import cv2
import numpy as np
from scipy.signal import correlate
# from skimage.registration import phase_cross_correlation # Alternative for phase correlation
import time
import math
import pantilthat as pth # Using actual pantilthat
from picamera2 import Picamera2 # Using actual Picamera2

# --- Configuration & Placeholders ---
# ROI Definition
ROI_X_START = 150
ROI_WIDTH = 400
# ROI_HEIGHT will be determined after camera setup and frame rotation

# Object detection parameters
OBJECT_THRESHOLD_VALUE = 30
MIN_OBJECT_HEIGHT = 5 # Minimum pixel height of a detected object segment

# PID Controller & Camera Optics Parameters
PIXEL_SIZE_MM = 0.00155  # For IMX477 sensor (1.55um)
FOCAL_LENGTH_MM = 25.0 # Updated focal length

# DEG_PER_PX will be calculated after ROI_HEIGHT is known
# PID Gains (Adjusted for less sensitivity)
PID_KP = 0.03 # Reduced Kp
PID_KI = 0.02  # Reduced Ki
PID_KD = 0.025  # Potentially reduced Kd
PID_TAU = 0.05 # Increased Tau for smoother derivative
PID_MAX_DT = 0.1

# TARGET_Y_IN_ROI will be calculated after ROI_HEIGHT is known

# Servo Configuration
SERVO_MIN_ANGLE = -90
SERVO_MAX_ANGLE = 90
INITIAL_TILT_ANGLE = -30.0 # Updated: Upwards tilt is negative
TILT_SERVO_CHANNEL = 2 # Standard tilt servo channel for PanTiltHAT

# Object Loss Handling
INTEGRAL_DAMPING_FACTOR_ON_LOSS = 0.95 # Factor to reduce integral by each frame when object is lost


# --- PID CONTROLLER (User's Code) ---
class PID:
    def __init__(self, kp, ki, kd, setpoint, deg_per_px,
                 tau=0.02, max_dt=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.deg_per_px = deg_per_px
        self.tau = tau # Time constant for derivative filter
        self.max_dt = max_dt
        self.prev_time = time.monotonic()
        self.integral = 0.0
        self.deriv = 0.0
        self.prev_measurement = None
        self.last_error = 0.0
        self.last_dt = 1e-6

    def update(self, measurement):
        now = time.monotonic()
        dt = max(1e-6, min(now - self.prev_time, self.max_dt))
        error = self.setpoint - measurement # Error in pixels

        # Proportional term
        P = self.kp * error

        # Integral term (accumulate error)
        self.integral += error * dt
        I = self.ki * self.integral

        # Derivative term (on measurement, with low-pass filter)
        if self.prev_measurement is None or dt == 0: # dt check for safety
            self.deriv = 0.0
        else:
            # Derivative of measurement
            raw_deriv = (measurement - self.prev_measurement) / dt # pixels/sec
            # Low-pass filter for derivative
            alpha = dt / (self.tau + dt) if (self.tau + dt) > 0 else 1.0 # Avoid division by zero
            self.deriv = alpha * raw_deriv + (1 - alpha) * self.deriv

        D = -self.kd * self.deriv # Negative feedback as derivative is on measurement

        output_px = P + I + D # Total correction in pixel units

        # Save state for next iteration
        self.prev_time = now
        self.prev_measurement = measurement
        self.last_error = error
        self.last_dt = dt

        return output_px * self.deg_per_px # Delta degrees

    def reset_integral(self):
        self.integral = 0.0
        print("PID Integral Reset")

    def dampen_integral(self, factor):
        self.integral *= factor


# --- Global state for camera tilt ---
current_tilt_angle = INITIAL_TILT_ANGLE

# --- Actuator Control Function ---
def control_camera_tilt(delta_deg_command, pid_controller):
    global current_tilt_angle

    # Optional: Clamp the maximum change in angle per step to prevent overly jerky movements
    # MAX_DELTA_DEG_PER_STEP = 2.0 # Example: limit to 2 degrees change per frame
    # delta_deg_command = np.clip(delta_deg_command, -MAX_DELTA_DEG_PER_STEP, MAX_DELTA_DEG_PER_STEP)

    prospective_tilt = current_tilt_angle + delta_deg_command

    # Anti-windup for servo limits
    if prospective_tilt > SERVO_MAX_ANGLE or prospective_tilt < SERVO_MIN_ANGLE:
        # If saturation occurs, reduce integral to prevent windup
        # This uses the pid_controller's stored last_error and last_dt
        # This specific anti-windup logic might need tuning or alternative approaches
        # A common way is to stop accumulating integral when output is saturated.
        # Here, we're retroactively adjusting based on the last error.
        pid_controller.integral -= pid_controller.last_error * pid_controller.last_dt

    current_tilt_angle -= delta_deg_command
    current_tilt_angle = max(SERVO_MIN_ANGLE, min(SERVO_MAX_ANGLE, current_tilt_angle))

    pth.tilt(int(round(current_tilt_angle))) # Standard PanTiltHAT tilt control


# --- Image Processing Functions ---
def extract_vertical_roi(frame, current_roi_x_start, current_roi_width):
    frame_h, frame_w = frame.shape[:2]
    x_start = max(0, min(ROI_X_START, frame_w - 1))
    x_end = max(0, min(ROI_X_START + ROI_WIDTH, frame_w))

    if x_start >= x_end :
        return np.array([])

    return frame[:, x_start:x_end]


def calculate_1d_projection(roi_gray):
    if roi_gray.size == 0 or roi_gray.shape[1] == 0: return np.array([])
    return np.mean(roi_gray, axis=1)

def estimate_camera_motion_1d_correlation(signal_t, signal_t_minus_1):
    if signal_t.size == 0 or signal_t_minus_1.size == 0 or signal_t.size != signal_t_minus_1.size:
        return 0
    if np.all(signal_t == signal_t[0]) or np.all(signal_t_minus_1 == signal_t_minus_1[0]):
        return 0
    correlation = correlate(signal_t, signal_t_minus_1, mode='same', method='fft')
    center_index = len(correlation) // 2
    peak_index = np.argmax(correlation)
    return center_index - peak_index

def align_1d_signal(signal, delta_y):
    if signal.size == 0: return np.array([])
    aligned_signal = np.roll(signal, int(delta_y))
    if delta_y > 0: aligned_signal[:int(delta_y)] = np.mean(signal) if signal.size > 0 else 0
    elif delta_y < 0: aligned_signal[int(delta_y):] = np.mean(signal) if signal.size > 0 else 0
    return aligned_signal

def detect_object_in_diff_1d(diff_signal, threshold_val, current_roi_height):
    if diff_signal.size == 0: return None
    potential_object_pixels = np.where(diff_signal > threshold_val)[0]
    if len(potential_object_pixels) >= MIN_OBJECT_HEIGHT:
        y_min = np.min(potential_object_pixels)
        y_max = np.max(potential_object_pixels)
        if (y_max - y_min + 1) < (current_roi_height / 1.5) :
             return (y_min, y_max)
    return None

# --- Main Loop ---
if __name__ == "__main__":
    picam2 = Picamera2()

    sensor_modes = picam2.sensor_modes
    if not sensor_modes:
        print("Error: No sensor modes available!")
        exit()
    fastmode = sensor_modes[0]

    capture_width, capture_height = fastmode['size']
    effective_frame_width = capture_height
    effective_frame_height = capture_width

    ROI_HEIGHT = effective_frame_height
    TARGET_Y_IN_ROI = ROI_HEIGHT // 2

    vfov_rad = 2 * math.atan((ROI_HEIGHT * PIXEL_SIZE_MM) / (2 * FOCAL_LENGTH_MM)) if FOCAL_LENGTH_MM > 0 else math.pi / 2
    vfov_deg = math.degrees(vfov_rad)
    DEG_PER_PX = (vfov_deg / ROI_HEIGHT) if ROI_HEIGHT > 0 else 0.03

    print(f"Selected sensor mode: {fastmode}")
    print(f"Capture size: {capture_width}x{capture_height}")
    print(f"Rotated frame size (WxH): {effective_frame_width}x{effective_frame_height}")
    print(f"ROI_HEIGHT set to: {ROI_HEIGHT}")

    camera_config = picam2.create_preview_configuration(
        sensor={'output_size': fastmode['size'], 'bit_depth': fastmode['bit_depth']},
        main={"format": "BGR888", "size": fastmode['size']},
        controls={"FrameDurationLimits": (8333, 8333)}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(1.0)

    pid_controller = PID(kp=PID_KP, ki=PID_KI, kd=PID_KD,
                         setpoint=TARGET_Y_IN_ROI,
                         deg_per_px=DEG_PER_PX,
                         tau=PID_TAU, max_dt=PID_MAX_DT)

    pth.servo_enable(TILT_SERVO_CHANNEL, True)
    pth.tilt(int(round(current_tilt_angle)))

    signal_t_minus_1 = None
    running = True

    cv2.namedWindow("Live View with Tracking", cv2.WINDOW_NORMAL)

    print("Starting main loop. Press 'q' in the Live View window or Ctrl+C in console to quit.")
    print(f"Picamera2 configured for ~{1e6/8333:.0f} FPS (requested).")
    print(f"Using ROI: XStart={ROI_X_START}, Width={ROI_WIDTH}, Height={ROI_HEIGHT}")
    print(f"Calculated Deg/Px: {DEG_PER_PX:.4f}")
    print(f"PID Setpoint (pixels): {pid_controller.setpoint}")
    print(f"Initial Tilt Angle: {current_tilt_angle} degrees")
    print(f"PID Gains: Kp={PID_KP}, Ki={PID_KI}, Kd={PID_KD}, Tau={PID_TAU}")


    loop_count = 0
    fps_calc_start_time = time.monotonic()
    display_fps = 0
    frames_object_not_detected = 0

    try:
        while running:
            frame_t_raw_bgr = picam2.capture_array("main")
            if frame_t_raw_bgr is None:
                print("Warning: Failed to capture frame.")
                time.sleep(0.01)
                continue

            frame_t_bgr = cv2.rotate(frame_t_raw_bgr, cv2.ROTATE_90_CLOCKWISE)
            frame_t_gray = cv2.cvtColor(frame_t_bgr, cv2.COLOR_BGR2GRAY)
            roi_t_gray = extract_vertical_roi(frame_t_gray, ROI_X_START, ROI_WIDTH)

            if roi_t_gray.size == 0 or roi_t_gray.shape[1] == 0:
                cv2.putText(frame_t_bgr, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame_t_bgr, "ROI ERROR", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Live View with Tracking", frame_t_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                time.sleep(0.01)
                continue

            signal_t = calculate_1d_projection(roi_t_gray)

            delta_y_cam_visual = 0
            if signal_t_minus_1 is not None and signal_t.size == signal_t_minus_1.size:
                delta_y_cam_visual = estimate_camera_motion_1d_correlation(signal_t, signal_t_minus_1)

            delta_y_cam = delta_y_cam_visual

            object_bounds_y = None
            y_object_current_for_pid = None

            if signal_t_minus_1 is not None and signal_t.size == signal_t_minus_1.size:
                signal_t_minus_1_aligned = align_1d_signal(signal_t_minus_1, delta_y_cam)
                diff_signal = np.abs(signal_t - signal_t_minus_1_aligned)
                object_bounds_y = detect_object_in_diff_1d(diff_signal, OBJECT_THRESHOLD_VALUE, ROI_HEIGHT)

            if object_bounds_y is not None:
                frames_object_not_detected = 0 # Reset counter
                y_min, y_max = object_bounds_y
                y_object_current_for_pid = (y_min + y_max) // 2
                delta_angle_output = pid_controller.update(y_object_current_for_pid)
                control_camera_tilt(delta_angle_output, pid_controller)

                draw_x_start = max(0, min(ROI_X_START, frame_t_bgr.shape[1] -1))
                draw_x_end = max(0, min(ROI_X_START + ROI_WIDTH, frame_t_bgr.shape[1]))
                if draw_x_start < draw_x_end:
                    cv2.rectangle(frame_t_bgr, (draw_x_start, y_min), (draw_x_end, y_max), (0, 0, 255), 2)
            else:
                # OBJECT NOT DETECTED
                frames_object_not_detected += 1
                # Dampen or reset integral to prevent runaway if object is lost
                if frames_object_not_detected > 5: # Start damping/resetting after a few frames of loss
                    pid_controller.dampen_integral(INTEGRAL_DAMPING_FACTOR_ON_LOSS)
                    # pid_controller.reset_integral() # Alternative: harsher reset
                # Hold current camera position (do not call control_camera_tilt)
                # Or, optionally, implement a slow return to a default position:
                # if current_tilt_angle != INITIAL_TILT_ANGLE:
                #     centering_error = INITIAL_TILT_ANGLE - current_tilt_angle
                #     # Small proportional control to slowly re-center
                #     centering_adjustment = np.clip(centering_error * 0.01, -0.1, 0.1)
                #     control_camera_tilt(centering_adjustment, pid_controller) # Careful with pid_controller here
                pass

            signal_t_minus_1 = signal_t.copy()
            loop_count +=1

            if (time.monotonic() - fps_calc_start_time) >= 1.0:
                display_fps = loop_count / (time.monotonic() - fps_calc_start_time)
                loop_count = 0
                fps_calc_start_time = time.monotonic()

            cv2.putText(frame_t_bgr, f"FPS: {display_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_t_bgr, f"Tilt: {current_tilt_angle:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if object_bounds_y is None:
                 cv2.putText(frame_t_bgr, "LOST", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            draw_roi_x_start = max(0, min(ROI_X_START, frame_t_bgr.shape[1] - 1))
            draw_roi_width_actual = max(0, min(ROI_WIDTH, frame_t_bgr.shape[1] - draw_roi_x_start))
            if draw_roi_width_actual > 0:
                 cv2.line(frame_t_bgr, (draw_roi_x_start, TARGET_Y_IN_ROI), (draw_roi_x_start + draw_roi_width_actual, TARGET_Y_IN_ROI), (0,255,255),1)


            cv2.imshow("Live View with Tracking", frame_t_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Exiting.")
    finally:
        print("[INFO] Disabling tilt servo and closing camera.")
        pth.servo_enable(TILT_SERVO_CHANNEL, False)
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("Exited.")
