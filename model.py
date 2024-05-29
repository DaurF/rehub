import mediapipe as mp

mp_pose = mp.solutions.pose


def check_exercise_1(data, landmarks):
    visible_side, left_elbow_angle, right_elbow_angle, left_hip_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle, = \
        data['visible_side'], data['left_elbow'], data['right_elbow'], data['left_hip'], data[
            'right_hip'], data['left_shoulder'], data['right_shoulder']

    if visible_side == 'r':
        for vis in [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility]:
            if vis < 0.5:
                raise RuntimeError("Can't see your body fully")

        if not (130 <= right_hip_angle <= 190):
            err_msg = 'Lay down straight.'
            if landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].visibility < 0.95:
                err_msg += " Might be because I can't see your right knee."
            raise RuntimeError(err_msg)
        if right_shoulder_angle > 120:
            raise RuntimeError('Bend your right shoulder')
        if not (right_elbow_angle <= 180):
            raise RuntimeError('Bend your right arm')
    elif visible_side == 'l':
        for vis in [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].visibility]:
            if vis < 0.5:
                raise RuntimeError("Can't see your body fully")

        if not (130 <= left_hip_angle <= 190):
            err_msg = 'Lay down straight.'
            if landmarks[mp_pose.PoseLandmark.LEFT_KNEE].visibility < 0.95:
                err_msg += " Might be because I can't see your left knee."
            raise RuntimeError(err_msg)
        if right_shoulder_angle > 120:
            raise RuntimeError('Bend your left shoulder')
        if not (left_elbow_angle <= 180):
            raise RuntimeError('Bend your left arm')
    return True


def check_exercise_2(data, landmarks):
    visible_side, left_knee, right_knee, left_elbow_angle, right_elbow_angle, left_hip_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle, = \
        data['visible_side'], data['left_knee'], data['right_knee'], data['left_elbow'], data['right_elbow'], data[
            'left_hip'], data['right_hip'], data['left_shoulder'], data['right_shoulder']

    for vis in [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility]:
        if vis < 0.5:
            raise RuntimeError("Can't see your body fully")

    if not (130 <= left_knee <= 180) or not (130 <= right_knee <= 180):
        raise RuntimeError('Put your feet on the floor')

    if not (95 <= left_hip_angle <= 175) or not (95 <= left_hip_angle <= 175):
        raise RuntimeError('Put your feet on the floor')

    if visible_side == 'r':
        if not (right_elbow_angle <= 180):
            raise RuntimeError('Lean on your right elbow and lift it')
        if not (right_shoulder_angle <= 45):
            raise RuntimeError('Lean on your right elbow and lift it')
    elif visible_side == 'l':
        if not (left_elbow_angle <= 180):
            raise RuntimeError('Lean on your left elbow and lift it')
        if not (left_shoulder_angle <= 45):
            raise RuntimeError('Lean on your left elbow and lift it')

    return True


def check_exercise_3(pose_estimation):
    # Extract angles from the pose estimation data
    shoulder_blade_floor_angle = pose_estimation.shoulder_blade_floor_angle
    upper_arm_torso_angle = pose_estimation.upper_arm_torso_angle
    elbow_angle = pose_estimation.elbow_angle
    shoulder_hand_angle = pose_estimation.shoulder_hand_angle

    # Check if the shoulder blades are retracted and depressed, maintaining a 20-degree angle with the floor
    if not (15 <= shoulder_blade_floor_angle <= 25):
        return False

    # Check if the arm is lifted to shoulder height, forming a 90-degree angle between upper arm and torso
    if not (85 <= upper_arm_torso_angle <= 95):
        return False

    # Check if the elbow forms a 90-degree angle when bent
    if not (85 <= elbow_angle <= 95):
        return False

    # Check if the arm is fully extended, forming a 180-degree angle from shoulder to hand
    if not (175 <= shoulder_hand_angle <= 185):
        return False

    # If all conditions are met, the exercise is done properly
    return True


def check_exercise_4(pose_estimation):
    # Extract angles from the pose estimation data
    knee_angle = pose_estimation.knee_angle
    hip_angle = pose_estimation.hip_angle
    thigh_torso_angle = pose_estimation.thigh_torso_angle
    hip_levelness = pose_estimation.hip_levelness

    # Check if the knees are bent at approximately 90 degrees
    if not (85 <= knee_angle <= 95):
        return False

    # Check if the hips form a straight line from shoulders to knees, maintaining an angle of approximately 180 degrees
    if not (175 <= hip_angle <= 185):
        return False

    # Check if the thigh and torso form a straight line during weight shift
    if not (175 <= thigh_torso_angle <= 185):
        return False

    # Check if the hips remain level throughout the exercise
    if not (175 <= hip_levelness <= 185):
        return False

    # If all conditions are met, the exercise is done properly
    return True


def check_exercise_5(pose_estimation):
    # Extract angles from the pose estimation data
    body_alignment_angle = pose_estimation.body_alignment_angle
    top_leg_bent_angle = pose_estimation.top_leg_bent_angle
    top_leg_straight_angle = pose_estimation.top_leg_straight_angle
    hip_levelness = pose_estimation.hip_levelness
    hip_shoulder_angle = pose_estimation.hip_shoulder_angle

    # Check if the head, hips, knees, and toes form a straight line (180 degrees)
    if not (175 <= body_alignment_angle <= 185):
        return False

    # Check if the top leg is bent at approximately 90 degrees
    if not (85 <= top_leg_bent_angle <= 95):
        return False

    # Check if the top leg is straightened back to 180 degrees
    if not (175 <= top_leg_straight_angle <= 185):
        return False

    # Check if the hips remain level, avoiding tilting or rotation
    if not (175 <= hip_levelness <= 185):
        return False

    # Check if the angle between hips and shoulders remains 180 degrees
    if not (175 <= hip_shoulder_angle <= 185):
        return False

    # If all conditions are met, the exercise is done properly
    return True


def check_exercise_6(pose_estimation):
    # Extract angles from the pose estimation data
    back_alignment_angle = pose_estimation.back_alignment_angle
    body_alignment_angle = pose_estimation.body_alignment_angle
    hip_levelness = pose_estimation.hip_levelness
    heel_ground_angle = pose_estimation.heel_ground_angle
    knee_bent_angle = pose_estimation.knee_bent_angle
    knee_straight_angle = pose_estimation.knee_straight_angle

    # Check if the back is flat against the ground (180 degrees)
    if not (175 <= back_alignment_angle <= 185):
        return False

    # Check if the hips, knees, and feet are aligned in a straight line (180 degrees)
    if not (175 <= body_alignment_angle <= 185):
        return False

    # Check if the hips stay level, avoiding any tilting or hiking
    if not (175 <= hip_levelness <= 185):
        return False

    # Check if the heel stays in constant contact with the ground (180 degrees)
    if not (175 <= heel_ground_angle <= 185):
        return False

    # Check if the knee achieves a fully extended position (180 degrees) when straightened
    if not (175 <= knee_straight_angle <= 185):
        return False

    # Check if the knee bends to approximately 90 degrees
    if not (85 <= knee_bent_angle <= 95):
        return False

    # If all conditions are met, the exercise is done properly
    return True


def check_exercise_7(pose_estimation):
    # Extract angles from the pose estimation data
    feet_alignment = pose_estimation.feet_alignment
    ankle_angle = pose_estimation.ankle_angle
    knee_angle = pose_estimation.knee_angle
    hip_angle = pose_estimation.hip_angle
    spine_angle = pose_estimation.spine_angle
    shoulder_hip_alignment = pose_estimation.shoulder_hip_alignment
    balance_stability = pose_estimation.balance_stability

    # Check if the feet are aligned in a heel-to-toe position
    if not (175 <= feet_alignment <= 185):
        return False

    # Check if the ankles maintain a 90-degree angle
    if not (85 <= ankle_angle <= 95):
        return False

    # Check if the knees maintain a 90-degree angle
    if not (85 <= knee_angle <= 95):
        return False

    # Check if the hips maintain a 90-degree angle
    if not (85 <= hip_angle <= 95):
        return False

    # Check if the spine is straight, forming a 180-degree angle from head to hips
    if not (175 <= spine_angle <= 185):
        return False

    # Check if the shoulders are aligned over the hips, forming a 180-degree alignment
    if not (175 <= shoulder_hip_alignment <= 185):
        return False

    # If all conditions are met, the exercise is done properly
    return True


def check_exercise_8(pose_estimation):
    # Extract angles from the pose estimation data
    ankle_angle = pose_estimation.ankle_angle
    knee_angle = pose_estimation.knee_angle
    hip_angle = pose_estimation.hip_angle
    spine_angle = pose_estimation.spine_angle
    shoulder_hip_alignment = pose_estimation.shoulder_hip_alignment

    # Check if the ankles maintain a 90-degree angle
    if not (85 <= ankle_angle <= 95):
        return False

    # Check if the knees maintain a 90-degree angle
    if not (85 <= knee_angle <= 95):
        return False

    # Check if the hips maintain a 90-degree angle
    if not (85 <= hip_angle <= 95):
        return False

    # Check if the spine is straight, forming a 180-degree alignment from head to feet
    if not (175 <= spine_angle <= 185):
        return False

    # Check if the shoulders are aligned over the hips, forming a 180-degree alignment
    if not (175 <= shoulder_hip_alignment <= 185):
        return False

    # If all conditions are met, the exercise is done properly
    return True
