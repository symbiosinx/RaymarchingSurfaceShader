Shader "Custom/Sinx/Raymarching" {
    Properties {
		[KeywordEnum(Cutout, Transparent)] _RenderMode("Rendering Mode", Float) = 0
		[Space]
        _Color("Color", Color) = (1,1,1,1)
        _MainTex("Albedo (RGB)", 2D) = "white" {}
		_NormalMap("Normal Map", 2D) = "bump" {}
        _Metallic("Metallic", Range(0,1)) = 0.0
        _Glossiness("Smoothness", Range(0,1)) = 0.5
		[Space]
		[Enum(Sphere, 0, Cube, 1, Torus, 2, Sponge, 3, Bulb, 4, Sierpinski, 5)] _Shape("Shape", Int) = 0
		_StepFactor("Step Factor", Range(0.05, 1)) = 1
		_MaxSteps("Max Ray Steps", Range(0, 2000)) = 100
		_MaxDist("Max Ray Distance", Range(0, 2000)) = 100
		_ContactThreshold("Contact Threshold", Range(0.00001, 0.1)) = 0.01
		_NormalSampleScale("Normal Sample Scale", Range(0.00001, 0.01)) = 0.01
		_IOR("Refractive Index", Range(1, 2)) = 1
    }
    SubShader {

        Tags { "Queue"="Geometry" "RenderType"="Opaque" }
        LOD 200

		Cull Back
		//ZTest LEqual

        CGPROGRAM

		#include "UnityPBSLighting.cginc"

        #pragma surface surf Spheretracing fullforwardshadows
        #pragma target 3.0

		float4x4 _CameraFrustrumCorners;
		float4x4 _CameraInvViewMatrix;
		sampler2D _CameraDepthTexture;

        fixed4 _Color;
        sampler2D _MainTex;
		sampler2D _NormalMap;
        half _Metallic;
        half _Glossiness;	

		int _Shape;
		float _MaxSteps;
		float _MaxDist;
		float _StepFactor;
		float _ContactThreshold;
		float _NormalSampleScale;
		float _IOR;

		float _FractalScale = 1.25;
		float _FractalRotationX = 0;
		float _FractalRotationY = 0;
		float _FractalRotationZ = 0;

		float3 surfacepoint;
		float depth;

        struct Input {
            float2 uv_MainTex;
			float2 uv_NormalMap;
			float3 viewDir;
			float4 screenPos;
			float3 worldPos;
        };

        UNITY_INSTANCING_BUFFER_START(Props)
        UNITY_INSTANCING_BUFFER_END(Props)
		
		float3 getcamviewdir(float2 uv) {
			
			float3 dir = lerp(
				lerp(_CameraFrustrumCorners[0].xyz, _CameraFrustrumCorners[1].xyz, uv.x), 
				lerp(_CameraFrustrumCorners[3].xyz, _CameraFrustrumCorners[2].xyz, uv.x),
			uv.y);
			dir = mul(_CameraInvViewMatrix, dir).xyz;
			return dir;
		}

        void surf (Input IN, inout SurfaceOutputStandard o) {
            // Albedo comes from a texture tinted by color
            fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
            o.Albedo = c.rgb;
			o.Normal = UnpackNormal(tex2D(_NormalMap, IN.uv_NormalMap));
            // Metallic and smoothness come from slider variables
            o.Metallic = _Metallic;
            o.Smoothness = _Glossiness;
			o.Alpha = 1;
            //o.Alpha = smoothstep(1, 0, 1-length(IN.uv_MainTex-.5));
			surfacepoint = IN.worldPos;
			float2 screenPos = IN.screenPos.xy / IN.screenPos.w;
			depth = Linear01Depth(tex2D(_CameraDepthTexture, screenPos).r) * _ProjectionParams.z * length(getcamviewdir(screenPos));
			depth = 10000;
			//depth = LinearEyeDepth(UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, screenPos)));
        }

		// Spheretracing functions
			struct ray {
				bool hit;
				float steps;
				float length;
				float3 origin;
				float3 direction;
				float closestdistance;
			};

			float repeat(float x, float m) {
				x += m * .5;
				float r = fmod(x, m);
				float o = float(r < 0 ? r + m : r) - m * .5;
				return o;
			}

			float2 repeat(float2 x, float2 m) {
				x += m * .5;
				float2 r = fmod(x, m);
				float2 o = float2(r.x < 0 ? r.x + m.x : r.x, r.y < 0 ? r.y + m.y : r.y) - m * .5;
				return o;
			}

			float3 repeat(float3 x, float3 m) {
				x += m * .5;
				float3 r = fmod(x, m);
				float3 o = float3(r.x < 0 ? r.x + m.x : r.x, r.y < 0 ? r.y + m.y : r.y, r.z < 0 ? r.z + m.z : r.z) - m * .5;
				return o;
			}

			float3 blend(float3 c1, float3 c2, float k) {
				return normalize(lerp(c1, c2, k))*lerp(length(c1), length(c2), k);
			}

			float smin(float a, float b, float k = .5) {
				float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
				return lerp(b, a, h) - k * h * (1.0 - h);
			}

			float4 smin(float4 a, float4 b, float k = .5) {
				float h = clamp(0.5 + 0.5 * (b.w - a.w) / k, 0.0, 1.0);
				return float4(blend(b.rgb, a.rgb, h), lerp(b.w, a.w, h) - k * h * (1.0 - h));
			}

			float smax(float a, float b, float k = .5) {
				float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0);
				return lerp(-b, -a, h) - k * h * (1.0 - h);
			}

			float4 smax(float4 a, float4 b, float k = .5) {
				float h = clamp(0.5 + 0.5 * (a.w - b.w) / k, 0.0, 1.0);
				return float4(lerp(b.rgb, a.rgb, h), -lerp(-b.w, -a.w, h) - k * h * (1.0 - h));
			}

			float3 rotate(float3 v, float3 a) {
				float3 c = cos(a);
				float3 s = sin(a);
				float3x3 mx = float3x3(1, 0, 0, 0, c.x, -s.x, 0, s.x, c.x);
				float3x3 my = float3x3(c.y, 0, s.y, 0, 1, 0, -s.y, 0, c.y);
				float3x3 mz = float3x3(c.z, -s.z, 0, s.z, c.z, 0, 0, 0, 1);
				return mul(mz, mul(my, mul(mx, v)));
			}

			float3 rotate(float3 v, float4 q) {
				float3 u = -q.xyz;
				return 2 * dot(u, v) * u + (q.w*q.w - dot(u, u)) * v + 2 * q.w * cross(u, v);
			}

			float remap(float x, float o1, float o2, float n1, float n2) {
				return (x - o1) / (o2 - o1) * (n2 - n1) + n1;
			}

			float rand(float n) {
				return frac(sin(n)*43758.5453);
			}

			float rand(float2 co){
				return frac(sin(dot(co.xy, float2(12.9898,78.233))) * 43758.5453);
			}

			float noise(float x) {
				float i = floor(x);
				float f = frac(x);

				float a = rand(i);
				float b = rand(i + 1);

				float u = smoothstep(0, 1, f);

				return lerp(a, b, u);
			}

			float noise(float2 st) {
				float2 i = floor(st);
				float2 f = frac(st);

				float a = rand(i);
				float b = rand(i + float2(1.0, 0.0));
				float c = rand(i + float2(0.0, 1.0));
				float d = rand(i + float2(1.0, 1.0));

				float2 u = smoothstep(0, 1, f);

				return lerp(a, b, u.x) +
				(c - a)* u.y * (1.0 - u.x) +
				(d - b) * u.x * u.y;
			}

			float noise(float3 x) {
				// The noise function returns a value in the range -1.0f -> 1.0f

				float3 p = floor(x);
				float3 f = frac(x);

				f = f*f*(3.0-2.0*f);
				float n = p.x + p.y*57.0 + 113.0*p.z;

				return lerp(lerp(lerp( rand(n+0.0), rand(n+1.0),f.x),
							lerp( rand(n+57.0), rand(n+58.0),f.x),f.y),
						lerp(lerp( rand(n+113.0), rand(n+114.0),f.x),
							lerp( rand(n+170.0), rand(n+171.0),f.x),f.y),f.z);
			}

			float4 sphere(float3 p, float r=1.0) {
				return float4(1, 1, 1, length(p) - r*.5);
			}

			float4 box(float3 p, float3 b = float3(1.0, 1.0, 1.0)) {
				b *= .5;
				float3 d = abs(p) - b;
				return float4(1, 1, 1, length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0));
			}

			float4 torus(float3 p, float2 t=float2(.5, .25)) {
				float2 q = float2(length(p.xy) - t.x, p.z);
				return float4(1, 1, 1, length(q) - t.y);
			}

			float4 cylinder(float3 p, float h=.5, float r=.5) {
				float2 d = abs(float2(length(p.xz),p.y)) - float2(h,r);
				return float4(1, 1, 1, min(max(d.x,d.y),0.0) + length(max(d,0.0)));
			}

			float4 mandelbulb(float3 p, float e=7, float iters=12, float bailout=10) {
				float3 z = p;
				float c = p;
				float dr = 1.0;
				float r = 0.0;
				float o = bailout;
				float o2 = bailout;
				float o3 = bailout;
				for (float i = 0; i < iters; i++) {
					r = length(z);
					o = min(o, length(z - float3(1, 0, 0)));
					o2 = min(o2, length(z - float3(0, 1, 0)));
					//o2 = max(o2, length(z - float3(0, 0, 0)));
					o3 = min(o3, length(z - float3(0, 0, 1)));
					if (r > bailout) break;

					// convert to polar coordinates
					float theta = acos(z.z / r);
					float phi = atan2(z.y, z.x);
					dr = pow(r, e - 1.0) * e * dr + 1.0;

					// scale and rotate the point
					float zr = pow(r, e);
					theta = theta * e;
					phi = phi * e;

					// convert back to cartesian coordinates
					z = zr * float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
					z += p;
				}
				return float4(o, o2, o3, .5 * log(r) * r / dr);
			}

			float4 sierpinski(float3 p) {
				float x = p.x; float y = p.y; float z = p.z;
				float r = x * x + y * y + z * z;
				float scale = _FractalScale;
				float bailout = 20;
				float o = bailout;
				float o2 = bailout;
				float o3 = bailout;
				float3 c;
				for (float i = 0; i < 15 && r < bailout; i++) {
					//Folding... These are some of the symmetry planes of the tetrahedron
					if (x + y < 0) { float x1 = -y;y = -x;x = x1; }
					if (x + z < 0) { float x1 = -z;z = -x;x = x1; }
					if (y + z < 0) { float y1 = -z;z = -y;y = y1; }

					c = float3(x, y, z);
					c = rotate(c, float3(_FractalRotationX, _FractalRotationY, _FractalRotationZ));
					x = c.x; y = c.y; z = c.z;

					//Stretches about the point [1,1,1]*(scale-1)/scale; The "(scale-1)/scale" is here in order to keep the size of the fractal constant wrt scale
					x = scale * x - (scale - 1); //equivalent to: x=scale*(x-cx); where cx=(scale-1)/scale;
					y = scale * y - (scale - 1);
					z = scale * z - (scale - 1);
					r = x * x + y * y + z * z;
					o = min(o, length(float3(x, y, z) - float3(1, 0, 0)));
					o2 = min(o2, length(float3(x, y, z) - float3(0, 1, 0)));
					o3 = min(o3, length(float3(x, y, z) - float3(0, 0, 1)));
				}
				return float4(o, o2, o3, (sqrt(r) - 2) * pow(scale, -i)); //the estimated distance
			}

			float4 menger(float3 p) {
				int n,iters=12;float t;
				float x = p.x, y = p.y, z = p.z;
				float o = 50; float o2 = 50; float o3 = 50;
				for(n=0;n<iters;n++){
					x=abs(x);y=abs(y);z=abs(z);
					if(x<y){t=x;x=y;y=t;}
					if(y<z){t=y;y=z;z=t;}
					if(x<y){t=x;x=y;y=t;}
					p = rotate(float3(x, y, z), float3(_FractalRotationX, _FractalRotationY, _FractalRotationZ));
					x = p.x; y = p.y; z = p.z;
					x=x*3.0-2.0;y=y*3.0-2.0;z=z*3.0-2.0;
					if(z<-1.0)z+=2.0;
					o = min(o, length(float3(x, y, z)   - float3(-1, 0, 0)));
					o2 = min(o2, length(float3(x, y, z) - float3(0, 1, 0)));
					o3 = min(o3, length(float3(x, y, z) - float3(0, 0, -1)));
				}
				return float4(o, o2, o3, (sqrt(x*x+y*y+z*z)-1.5)*pow(3.0,-(float)iters));
			}

			float4 scene(float3 p) {
				// p.xz = repeat(p.xz, 5);
				float4 tor = torus(rotate(p, float3(0, UNITY_PI*.5, 0)), float2(.3, .15));
				tor.rgb = float3(1, 1, 1);
				float4 sph = sphere(p, 1);
				float4 b = box(p, 1);
				float4 mandel = mandelbulb(p*2.25, remap(sin(_Time.x*5), -1, 1, 2, 12));
				mandel.rgb *= .75;
				// mandel.rgb = 1;
				float4 meng = menger(p*2.25);
				meng.rgb *= .5;
				// meng.rgb = 1;
				float4 sier = sierpinski(p*2.5);
				float4 shapes[6] = {
					sph,
					b,
					tor,
					meng,
					mandel,
					sier
				};
				return shapes[_Shape];
			}

			ray spheretrace(float3 ro, float3 rd, float depth=2000, float steps=2000) {
				depth = min(depth, _MaxDist);
				steps = min(steps, _MaxSteps);
				float dm = 0;
				bool hit;
				float cd = scene(ro).w;
				for (int i = 0; i < steps; i++) {
					float3 cp = ro + rd * dm;
					float dts = scene(cp).w;
					cd = min(cd, dts);
					dm += dts * _StepFactor;
					//_ContactThreshold = dm*.0025;
					if (dts < _ContactThreshold) {
						hit = true;
						break;
					}
					if (dm > depth) {
						break;
					}
					if (dm < 0) {
						hit = true;
						break;
					}
				}
				ray r;
				r.hit = hit;
				r.steps = i;
				r.length = dm;
				r.origin = ro;
				r.direction = rd;
				r.closestdistance = cd;
				return r;
			}

			float3 getnormalraw(float3 p, float s = 0.) {
				float2 e = float2(max(s, _NormalSampleScale), 0);

				return (float3(
					scene(p + e.xyy).w - scene(p - e.xyy).w,
					scene(p + e.yxy).w - scene(p - e.yxy).w,
					scene(p + e.yyx).w - scene(p - e.yyx).w
				));

			}

			float3 getnormal(float3 p, float s=0.) {
				return normalize(getnormalraw(p, s));
			}

			float getAO(float3 n) {
				return clamp(length(n/_NormalSampleScale), 0, 1);
			}

			float getshadow(float3 p, float3 n, float3 l, float k=16, float depth=2000, float steps=2000) {
				depth = min(depth, _MaxDist);
				steps = min(steps, _MaxSteps);
				float dm = 0;
				float res = 1;
				p += n * _ContactThreshold*2;
				for (int i = 0; i < steps; i++) {
					float3 cp = p + l * dm;
					float dts = scene(cp).w;
					res = min(res, k*dts/dm);
					dm += abs(dts) * _StepFactor;
					//_ContactThreshold = dm*.0025;
					if (abs(dts) < _ContactThreshold) {
						res = 0;
						break;
					}
					if (dm > depth) {
						break;
					}
				}

				return res;
			}
		// End of Spheretracing Functions

		float3 hitpoint;
		float3 worldnormal;
		float occlusion;
		float hit;
		float steps;
		float closestdistance;

		void LightingSpheretracing_GI(SurfaceOutputStandard s, UnityGIInput data, inout UnityGI gi) {

			float3 origin = mul(unity_ObjectToWorld, float4(0, 0, 0, 1));
			float3 rayorigin = surfacepoint - origin;
			float3 view = refract(-data.worldViewDir, s.Normal, _IOR);
			ray r = spheretrace(rayorigin, view);
			if (!r.hit) discard;
			float3 objecthitpoint = rayorigin + view * r.length;
			float3 worldhitpoint = origin + objecthitpoint;
			clip(depth - length(worldhitpoint - _WorldSpaceCameraPos));

			float3 rawnormal = getnormalraw(objecthitpoint);
			float3 normal = normalize(rawnormal);
			float shadow = getshadow(objecthitpoint, normal, gi.light.dir);
			//shadow = 1;
			float diffuse = dot(normal, data.light.dir)*.5+.5;
			float AO = getAO(rawnormal);

			if (r.length <= 0) {
				shadow = 1;
				normal = s.Normal;
			}

			hitpoint = objecthitpoint;
			worldnormal = normal;
			occlusion = diffuse * AO;
			hit = r.hit;
			steps = r.steps;
			closestdistance = r.closestdistance;

			data.worldPos = worldhitpoint;
			data.atten = shadow;
			data.ambient = SHEvalLinearL2(float4(normal, 1));
			s.Normal = normal;

			LightingStandard_GI(s, data, gi);
		}

		inline fixed4 LightingSpheretracing(SurfaceOutputStandard s, fixed3 viewDir, UnityGI gi) {
			fixed4 pbr = LightingStandard(s, viewDir, gi);
			pbr = 0;

			s.Normal = worldnormal;
			s.Alpha = hit ? 1 : 0;
			//gi.light.color = gi.light.color * occlusion;

			fixed4 col = 1;
			fixed4 rmpbr = LightingStandard(s, viewDir, gi);
			rmpbr.rgb *= scene(hitpoint).rgb;
			
			col.rgb = pbr.rgb * pbr.a + rmpbr.rgb * rmpbr.a * (1 - pbr.a);
			col.a = 1 - ((1 - rmpbr.a) * (1 - pbr.a));

			return col;
		}

        ENDCG
		//UsePass "Legacy Shaders/VertexLit/SHADOWCASTER"
    }
    FallBack "Transparent/Cutout/VertexLit"
}
