#include "camera.h"
#include "glm/gtx/transform.hpp"
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/quaternion.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Camera::Camera()
{
    pitch = 0;
    yaw = 0;

    cameraPosition = { 0.f, 6.f, 10.f };
    cameraForward = { 0.f ,0.f, -1.f };
    cameraUp = { 0.f, 1.f, 0.f };
    cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));

    //proj
    projection = glm::perspective(glm::radians(70.f), 1700.f / 900.f, 0.1f, 200.f);
    projection[1][1] *= -1;

    UpdateCameraMatrices();
}

void Camera::MoveCamera(glm::vec3 direction)
{
    cameraPosition += direction * movementSpeed;
    UpdateCameraMatrices();
}

void Camera::RotateCamera(float inPitch, float inYaw)
{
    const float halfPI = M_PI / 2.0f - 0.1;

    if (pitch > halfPI && inPitch > 0)
    {
        inPitch = 0;
    }
    if (pitch < -halfPI && inPitch < 0)
    {
        inPitch = 0;
    }

    pitch += inPitch;
    yaw += inYaw;

    //calculate new forward vector given the new pitch and yaw
    glm::vec3 direction;
    direction.x = sin(yaw) * -cos(pitch);
    direction.y = sin(pitch);
    direction.z = cos(yaw) * -cos(pitch);
    cameraForward = glm::normalize(direction);
    cameraRight = glm::normalize(glm::cross(cameraForward, cameraUp));

    UpdateCameraMatrices();
}

void Camera::UpdateCameraMatrices()
{
    view = glm::lookAt(cameraPosition, cameraPosition + cameraForward, cameraUp);
    viewProjection = projection * view;
}
