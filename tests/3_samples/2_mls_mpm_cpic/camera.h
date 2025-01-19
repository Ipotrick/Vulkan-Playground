#pragma once

#ifndef CAMERA_H
#define CAMERA_H

#define GLM_ENABLE_EXPERIMENTAL
#include <array>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

const glm::vec3 INIT_FORWARD = {0, -1.25, -1};
const glm::vec3 INIT_CAMERA_POS = {0.5, 1.25, 1.5};
const glm::vec3 INIT_CAMERA_UP = {0, 1, 0};

const float INIT_CAMERA_FOV = 45.0f;
const unsigned INIT_CAMERA_WIDTH = 800;
const unsigned INIT_CAMERA_HEIGHT = 600;
const float INIT_CAMERA_NEAR = 0.001f;
const float INIT_CAMERA_FAR = 1000.0f;
const float CAMERA_SPEED = 0.01f;
const float MOUSE_SENSITIVITY = 0.005f;
const float SPEED_UP_MULTIPLIER = 10.0f;
const float CAMERA_DEF_FOCUS_DIST_MIN = 0.0f;
const float CAMERA_DEF_FOCUS_DIST_MAX = 100.0f;
const float CAMERA_DEF_FOCUS_ANGLE_MIN = 0.0f;
const float CAMERA_DEF_FOCUS_ANGLE_MAX = 5.0f;

const unsigned int TRIPPLE_BUFFER = 3;

// const glm::vec3 RIGHT_DIRECTION = {1.0f, 0.0f, 0.0f};

typedef struct camera {
    glm::vec3 forward;
    glm::vec3 position;
    glm::vec3 center;
    glm::vec3 up;
    float fov;
    unsigned int width;
    unsigned int height;
    float _near;
    float _far;
    // glm::mat4 view;
    // glm::mat4 projection;
    // glm::mat4 view_projection;
    // glm::mat4 inverse_view;
    glm::vec2 last_mouse_pos;
    float speed;
    bool mouse_left_press;
    bool mouse_middle_pressed;
    std::array<bool, TRIPPLE_BUFFER> moved;
    unsigned int frame_index;
    bool shift_status;
    float defocus_angle;
    float focus_dist;
} camera;


void reset_camera(camera& cam) {
    cam.forward = INIT_FORWARD;
    cam.position = INIT_CAMERA_POS;
    cam.up = INIT_CAMERA_UP;
    cam.fov = INIT_CAMERA_FOV;
    cam.width = INIT_CAMERA_WIDTH;
    cam.height = INIT_CAMERA_HEIGHT;
    cam._near = INIT_CAMERA_NEAR;
    cam._far = INIT_CAMERA_FAR;
    cam.speed = CAMERA_SPEED;
    cam.last_mouse_pos = glm::vec2(0, 0);
    cam.mouse_left_press = false;
    cam.mouse_middle_pressed = false;
    cam.moved[0] = cam.moved[1] = cam.moved[2] = true;
    cam.frame_index = 0;
    cam.shift_status = false;
    cam.defocus_angle = CAMERA_DEF_FOCUS_ANGLE_MIN;
    cam.focus_dist = CAMERA_DEF_FOCUS_DIST_MIN;
}

const glm::mat4 get_view_matrix(const camera& cam) {
    return glm::lookAt(cam.position, cam.position + cam.forward, cam.up);
}

const glm::mat4 get_projection_matrix(const camera& cam) {
    return glm::perspectiveRH_ZO(cam.fov, static_cast<float>(cam.width) / static_cast<float>(cam.height) , cam._near, cam._far);
}

const glm::mat4 get_view_projection_matrix(const camera& cam) {
    return get_projection_matrix(cam) * get_view_matrix(cam);
}

const glm::mat4 get_inverse_view_matrix(const camera& cam) {
    return glm::inverse(get_view_matrix(cam));
}

const glm::mat4 get_inverse_projection_matrix(const camera& cam) {
    return glm::inverse(get_projection_matrix(cam));
}

const glm::mat4 get_inverse_view_projection_matrix(const camera& cam) {
    return glm::inverse(get_view_projection_matrix(cam));
}

const glm::vec3 camera_get_direction(const camera& cam) {
    return cam.forward;
}

const glm::vec3 camera_get_right(const camera& cam) {
    return glm::normalize(glm::cross(camera_get_direction(cam), cam.up));
}

const glm::vec3 camera_get_up(const camera& cam) {
    return glm::normalize(glm::cross(camera_get_right(cam), camera_get_direction(cam)));
}

const glm::vec3& camera_get_position(const camera& cam) {
    return cam.position;
}

const glm::vec3& camera_get_center(const camera& cam) {
    return cam.center;
}

const glm::vec3& camera_get_up_vector(const camera& cam) {
    return cam.up;
}

const float& camera_get_fov(const camera& cam) {
    return cam.fov;
}

float camera_get_aspect(const camera& cam) {
    return static_cast<float>(cam.width) / static_cast<float>(cam.height);
}

const unsigned int& camera_get_width(const camera& cam) {
    return cam.width;
}

const unsigned int& camera_get_height(const camera& cam) {
    return cam.height;
}

const float& camera_get__near(const camera& cam) {
    return cam._near;
}

const float& camera_get_far(const camera& cam) {
    return cam._far;
}

const float& camera_get_speed(const camera& cam) {
    return cam.speed;
}

const bool& camera_get_moved(const camera& cam) {
    return cam.moved[cam.frame_index];
}

const glm::vec2& camera_get_last_mouse_pos(const camera& cam) {
    return cam.last_mouse_pos;
}

const bool& camera_get_mouse_left_press(const camera& cam) {
    return cam.mouse_left_press;
}

const bool& camera_get_mouse_middle_pressed(const camera& cam) {
    return cam.mouse_middle_pressed;
}

const bool& camera_get_shift_status(const camera& cam) {
    return cam.shift_status;
}

const float& camera_get_defocus_angle(const camera& cam) {
    return cam.defocus_angle;
}

const float& camera_get_focus_dist(const camera& cam) {
    return cam.focus_dist;
}





void move_camera(camera& cam, const glm::vec3& direction) {
    cam.position += direction * cam.speed * (cam.shift_status ? SPEED_UP_MULTIPLIER : 1.0f);
    cam.moved[0] = cam.moved[1] = cam.moved[2] = true;
}

void move_camera_forward(camera& cam) {
    move_camera(cam,  camera_get_direction(cam));
}

void move_camera_backward(camera& cam) {
    move_camera(cam, -camera_get_direction(cam));
}

void move_camera_right(camera& cam) {
    move_camera(cam,  camera_get_right(cam));
}

void move_camera_left(camera& cam) {
    move_camera(cam, -camera_get_right(cam));
}

void move_camera_up(camera& cam) {
    move_camera(cam, camera_get_up(cam));
}

void move_camera_down(camera& cam) {
    move_camera(cam, -camera_get_up(cam));
}

void camera_set_position(camera& cam, const glm::vec3& position) {
    cam.position = position;
}

void camera_set_center(camera& cam, const glm::vec3& center) {
    cam.center = center;
}

void camera_set_up_vector(camera& cam, const glm::vec3& up) {
    cam.up = up;
}

void camera_set_fov(camera& cam, float fov) {
    cam.fov = fov;
}

void camera_set_aspect(camera& cam, unsigned int width, unsigned int height) {
    cam.width = width;
    cam.height = height;
}

void camera_set_near(camera& cam, float near_plane) {
    cam._near = near_plane;
}

void camera_set_far(camera& cam, float far_plane) {
    cam._far = far_plane;
}

void camera_shift_pressed(camera& cam) {
    cam.shift_status = true;
}

void camera_shift_released(camera& cam) {
    cam.shift_status = false;
}

void camera_set_speed(camera& cam, float speed) {
    cam.speed = speed;
}

void camera_set_moved(camera& cam) {
    cam.moved[0] = cam.moved[1] = cam.moved[2] = true;
}

void camera_reset_moved(camera& cam) {
    cam.moved[cam.frame_index] = false;
    cam.frame_index = (cam.frame_index + 1) % TRIPPLE_BUFFER;
}

void camera_set_last_mouse_pos(camera& cam, const glm::vec2& last_mouse_pos) {
    cam.last_mouse_pos = last_mouse_pos;
}

void camera_set_mouse_left_press(camera& cam, bool mouse_left_press) {
    cam.mouse_left_press = mouse_left_press;
}

void camera_set_mouse_middle_pressed(camera& cam, bool mouse_middle_pressed) {
    cam.mouse_middle_pressed = mouse_middle_pressed;
}

void camera_set_defocus_angle(camera& cam, float defocus_angle) {
    cam.defocus_angle = std::max(CAMERA_DEF_FOCUS_ANGLE_MIN, std::min(CAMERA_DEF_FOCUS_ANGLE_MAX, defocus_angle));
}

void camera_set_focus_dist(camera& cam, float focus_dist) {
    cam.focus_dist = std::max(CAMERA_DEF_FOCUS_DIST_MIN, std::min(CAMERA_DEF_FOCUS_DIST_MAX, focus_dist));
}

// rotate camera around its center
void rotate_camera(camera& cam, float currentX, float currentY)
{
    glm::vec3 up_direction = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 right_direction = glm::cross(cam.forward, up_direction);

    // check if last mouse position is set
    if (cam.last_mouse_pos.x == 0.0f && cam.last_mouse_pos.y == 0.0f)
    {
        camera_set_last_mouse_pos(cam, glm::vec2(currentX, currentY));
        return;
    }

    // Calculate the change in mouse position
    float deltaX = (currentX - cam.last_mouse_pos.x);
    float deltaY = (currentY - cam.last_mouse_pos.y);


    camera_set_last_mouse_pos(cam, glm::vec2(currentX, currentY));


    // Rotation
	if (deltaX != 0.0f || deltaY != 0.0f)
	{
		float pitch_delta = deltaY * MOUSE_SENSITIVITY;
		float yaw_delta = deltaX * MOUSE_SENSITIVITY;

		glm::quat q = glm::normalize(glm::cross(glm::angleAxis(-pitch_delta, right_direction),
			glm::angleAxis(-yaw_delta, glm::vec3(0.f, 1.0f, 0.0f))));
		cam.forward = glm::rotate(q, cam.forward);

        camera_set_moved(cam);
	}
}

void rotate_camera_yaw(camera& cam, float yaw) {
    rotate_camera(cam, yaw, 0.0f);
}

void rotate_camera_pitch(camera& cam, float pitch) {
    rotate_camera(cam, 0.0f, pitch);
}

void camera_set_mouse_delta(camera& cam, const glm::vec2& mouse_delta) {
    if(cam.mouse_left_press) {
        rotate_camera(cam, mouse_delta.x, mouse_delta.y);
    }
}

#endif // CAMERA_H