
#include <vk_types.h>

#include "glm/glm.hpp"

class Camera {
public:
	Camera();
	float movementSpeed = 0.1f;
	float radAngleIncrement = 0.03f;
	glm::mat4 view;
	glm::mat4 projection;
	glm::mat4 viewProjection;

	//new camera
	glm::vec3 cameraPosition;
	glm::vec3 cameraForward;
	glm::vec3 cameraUp;
	glm::vec3 cameraRight;

	float pitch;
	float yaw;

	void MoveCamera(glm::vec3 direction);
	void RotateCamera(float pitch, float yaw);

	void UpdateCameraMatrices();

};
