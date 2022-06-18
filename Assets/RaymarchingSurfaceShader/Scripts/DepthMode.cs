using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Camera))]
[ExecuteInEditMode]
[ImageEffectAllowedInSceneView]
public class DepthMode : MonoBehaviour {
	
	public DepthTextureMode depthTextureMode = DepthTextureMode.Depth;

    void Start() {
		
	}

	Matrix4x4 GetFrustumCorners(Camera cam) {
    	float camFov = cam.fieldOfView;
    	float camAspect = cam.aspect;

    	Matrix4x4 frustumCorners = Matrix4x4.identity;

    	float fovWHalf = camFov * 0.5f;
		
    	float tan_fov = Mathf.Tan(fovWHalf * Mathf.Deg2Rad);

    	Vector3 toRight = Vector3.right * tan_fov * camAspect;
    	Vector3 toTop = Vector3.up * tan_fov;

    	Vector3 topLeft = (-Vector3.forward - toRight + toTop);
    	Vector3 topRight = (-Vector3.forward + toRight + toTop);
    	Vector3 bottomRight = (-Vector3.forward + toRight - toTop);
    	Vector3 bottomLeft = (-Vector3.forward - toRight - toTop);

    	frustumCorners.SetRow(0, topLeft);
    	frustumCorners.SetRow(1, topRight);
    	frustumCorners.SetRow(2, bottomRight);
    	frustumCorners.SetRow(3, bottomLeft);
		
    	return frustumCorners;
	}

	void Update() {
	}

	void OnRenderImage(RenderTexture src, RenderTexture dest) {
		Camera cam = GetComponent<Camera>();
		cam.depthTextureMode = cam.depthTextureMode | DepthTextureMode.MotionVectors;
		Shader.SetGlobalMatrix("_CameraFrustrumCorners", GetFrustumCorners(cam));
		Shader.SetGlobalMatrix("_CameraInvViewMatrix", cam.cameraToWorldMatrix);
		Graphics.Blit(src, dest);		
	}
}
