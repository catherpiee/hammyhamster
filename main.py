import math
import os
from collections import Counter, deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    HandLandmarker,
    HandLandmarkerOptions,
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MEME_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(MEME_DIR, "images")
HAND_MODEL_PATH = os.path.join(MEME_DIR, "hand_landmarker.task")
FACE_MODEL_PATH = os.path.join(MEME_DIR, "face_landmarker.task")
POSE_MODEL_PATH = os.path.join(MEME_DIR, "pose_landmarker.task")

MEMES = {
    "default": "pokerham.jpg",
    "thumbs_up": "thumb.jpg",
    "thumbs_down": "thumbs down.jpg",
    "side_eye": "sideyee.jpg",
    "fist_by_head": "happylollypop.webp",
    "two_hands": "2 arms out .jpg",
    "glasses": "discord mod.jpg",
    "bicep": "bicep.jpg",
    "cross_arms": "cross arms .jpg",
    "finger_mouth": "one finger mouth .jpg",
    "nerd": "nerd.jpg",
}

WINDOW_W, WINDOW_H = 640, 640

# Tuning knobs for the "glitchy" behavior.
YAW_THRESHOLD_DEG = 18       # head turn needed to trigger side_eye
VOTE_WINDOW = 12             # frames of history considered
VOTE_MAJORITY = 7            # votes needed within the window to switch memes
GLASSES_NEAR_FACE_DIST = 0.28   # how close (normalized) a pinch must be to the face to count as "glasses"
MOUTH_NEAR_DIST = 0.14          # how close (normalized) the index fingertip must be to the mouth
ELBOW_BEND_MAX_DEG = 100        # elbow angle (shoulder-elbow-wrist) below this counts as "bent" for bicep
POSE_VISIBILITY_MIN = 0.5       # minimum pose-landmark visibility to trust a joint

# Face-mesh landmark id used as a stand-in for the mouth's position.
MOUTH_LANDMARK = 13

meme_images = {}
for key, filename in MEMES.items():
    img = cv2.imread(os.path.join(IMAGES_DIR, filename))
    meme_images[key] = cv2.resize(img, (WINDOW_W, WINDOW_H))

# (tip landmark id, base/knuckle landmark id) per non-thumb finger,
# wrist-relative so it works regardless of how the hand is rotated in frame.
FINGER_JOINTS = [(8, 5), (12, 9), (16, 13), (20, 17)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# BlazePose landmark ids used for the body-pose gestures.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24


def _dist(a, b):
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def fingers_up(landmarks):
    wrist = landmarks[0]

    # Measuring the thumb against palm-center distance only works well when
    # the palm faces the camera flat-on. A thumbs-up fist is rotated ~90°
    # from that, and in that pose the thumb tip barely moves laterally away
    # from palm center even fully extended, so it kept reading as curled.
    # The pinky-MCP (17) is on the far side of the hand from the thumb in
    # every common rotation, so tip-to-pinky-base distance grows a lot
    # whenever the thumb sticks out — sideways *or* straight up.
    pinky_base = landmarks[17]
    thumb_extended = _dist(landmarks[4], pinky_base) > _dist(landmarks[2], pinky_base) * 1.1
    fingers = [1 if thumb_extended else 0]

    for tip_id, base_id in FINGER_JOINTS:
        extended = _dist(wrist, landmarks[tip_id]) > _dist(wrist, landmarks[base_id]) * 1.15
        fingers.append(1 if extended else 0)
    return fingers


def classify_single_hand(fingers):
    thumb, index, middle, ring, pinky = fingers
    four_curled = not (index or middle or ring or pinky)

    # Thumb curl is the least reliable read (it depends on hand rotation),
    # so lead with the other four fingers: if they're all curled, it's a
    # fist (or thumbs-up if the thumb is *also* clearly sticking out).
    # Checking thumb-extended first used to misclassify a fist held
    # sideways next to the head as thumbs-up.
    if four_curled:
        return "thumbs_up" if thumb else "fist"
    if index and middle and ring and pinky and thumb:
        return "open_palm"
    if index and not middle and not ring and not pinky:
        return "pointer"
    return None


def thumb_dy_ratio(landmarks):
    # Wrist-relative vertical offset of the thumb tip, scaled by hand size.
    # Image y grows downward, so a positive ratio means the thumb tip sits
    # below the wrist (pointing down); negative means above (pointing up).
    scale = _dist(landmarks[0], landmarks[9])
    if scale < 1e-6:
        return 0.0
    return (landmarks[4].y - landmarks[0].y) / scale


def thumb_points_down(landmarks):
    # 0.35 margin (up from an earlier, twitchier 0.25) — flickered between
    # up/down on an upright thumb because normal hand-tracking jitter was
    # enough to cross a tighter line.
    return thumb_dy_ratio(landmarks) > 0.35


def pinch_dist_ratio(landmarks):
    # Thumb-tip-to-index-tip distance, scaled by wrist-to-middle-MCP
    # distance so it works regardless of hand size or distance from camera.
    scale = _dist(landmarks[0], landmarks[9])
    if scale < 1e-6:
        return 999.0
    return _dist(landmarks[4], landmarks[8]) / scale


def is_pinch(landmarks):
    # A raw thumb-to-index distance threshold alone can't tell a real
    # pinch apart from a closed fist: a fist naturally rests the thumb
    # near the curled fingers too, so loosening the threshold enough to
    # catch a real pinch (0.6) also caught fists, and tightening it enough
    # to exclude fists (0.4) missed real pinches. The actual shape
    # difference is that a pinch brings the thumb close to the index tip
    # *specifically* — noticeably closer than to the middle tip — while a
    # fist's thumb sits near both about equally.
    scale = _dist(landmarks[0], landmarks[9])
    if scale < 1e-6:
        return False
    thumb_index = _dist(landmarks[4], landmarks[8])
    thumb_middle = _dist(landmarks[4], landmarks[12])
    return thumb_index < scale * 0.5 and thumb_index < thumb_middle * 0.7


def _pose_visible(landmark):
    return getattr(landmark, "visibility", 1.0) >= POSE_VISIBILITY_MIN


def _elbow_angle_degrees(shoulder, elbow, wrist):
    v1 = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    v2 = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def bicep_signals(pose_landmarks):
    # Per-arm (angle_deg, wrist_above_shoulder, elbow_out_from_body) for
    # whichever arm is visible, or None if neither is. Exposed separately
    # from detect_bicep so the debug overlay can show the raw numbers.
    if not pose_landmarks:
        return None
    best = None
    for shoulder_i, elbow_i, wrist_i in ((LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),
                                          (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)):
        shoulder, elbow, wrist = pose_landmarks[shoulder_i], pose_landmarks[elbow_i], pose_landmarks[wrist_i]
        if not (_pose_visible(shoulder) and _pose_visible(elbow) and _pose_visible(wrist)):
            continue
        angle = _elbow_angle_degrees(shoulder, elbow, wrist)
        if angle is None:
            continue
        wrist_above = shoulder.y - wrist.y   # positive = wrist above shoulder
        elbow_out = abs(elbow.x - shoulder.x)  # positive = elbow away from torso, to the side
        if best is None or angle < best[0]:
            best = (angle, wrist_above, elbow_out)
    return best


def detect_bicep(pose_landmarks):
    # Bent elbow, wrist clearly ABOVE shoulder height (not just "not much
    # below" — the old margin let a thumbs-down held at chest/shoulder
    # height with a bent elbow satisfy it too), and the elbow held out away
    # from the torso, which a flex has and a thumbs-down/up in front of the
    # body doesn't.
    signals = bicep_signals(pose_landmarks)
    if signals is None:
        return False
    angle, wrist_above, elbow_out = signals
    return angle < ELBOW_BEND_MAX_DEG and wrist_above > 0.06 and elbow_out > 0.06


def detect_cross_arms(pose_landmarks):
    # Both wrists tucked close together at chest height. Real crossed arms
    # usually hide the hands under the armpits, where the hand landmarker
    # sees nothing at all — pose landmarks track the wrist joint from the
    # arm's silhouette regardless, so this doesn't depend on the hands
    # being visible.
    if not pose_landmarks:
        return False
    l_wrist, r_wrist = pose_landmarks[LEFT_WRIST], pose_landmarks[RIGHT_WRIST]
    l_shoulder, r_shoulder = pose_landmarks[LEFT_SHOULDER], pose_landmarks[RIGHT_SHOULDER]
    if not (_pose_visible(l_wrist) and _pose_visible(r_wrist)
            and _pose_visible(l_shoulder) and _pose_visible(r_shoulder)):
        return False

    l_hip, r_hip = pose_landmarks[LEFT_HIP], pose_landmarks[RIGHT_HIP]
    chest_top = min(l_shoulder.y, r_shoulder.y)
    if _pose_visible(l_hip) and _pose_visible(r_hip):
        chest_bottom = max(l_hip.y, r_hip.y)
    else:
        chest_bottom = chest_top + 0.35

    wrists_close = _dist(l_wrist, r_wrist) < 0.18
    avg_wrist_y = (l_wrist.y + r_wrist.y) / 2
    at_chest_height = chest_top < avg_wrist_y < chest_bottom
    return wrists_close and at_chest_height


def landmarks_center(landmarks):
    x = sum(p.x for p in landmarks) / len(landmarks)
    y = sum(p.y for p in landmarks) / len(landmarks)
    return np.array([x, y])


def head_yaw_degrees(transformation_matrix):
    # transformation_matrix is a 4x4 rotation+translation matrix from
    # FaceLandmarker. R[0][2] tracks how much the face has rotated about the
    # vertical axis (left/right turn); asin of it gives the yaw angle.
    r02 = transformation_matrix[0][2]
    return math.degrees(math.asin(max(-1.0, min(1.0, r02))))


def classify_gesture(hand_result, face_result, pose_result):
    hand_landmarks_list = hand_result.hand_landmarks
    pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None

    head_center = np.array([0.5, 0.3])
    mouth_point = None
    yaw_deg = None
    if face_result.face_landmarks:
        face_landmarks = face_result.face_landmarks[0]
        head_center = landmarks_center(face_landmarks)
        mouth_point = np.array([face_landmarks[MOUTH_LANDMARK].x, face_landmarks[MOUTH_LANDMARK].y])
    if face_result.facial_transformation_matrixes:
        yaw_deg = head_yaw_degrees(face_result.facial_transformation_matrixes[0])

    # Per-hand shape/position checks run BEFORE the body-pose (cross_arms/
    # bicep) and "two hands visible" fallbacks below, not after. A normal
    # thumbs-up or fist-beside-head naturally has a bent elbow with the
    # wrist raised near/above shoulder height too, so checking bicep's pose
    # geometry first used to swallow both of those into "bicep" before
    # their hand shape was ever read. Deciding by hand shape first — when a
    # hand is clearly readable — avoids that collision; the pose-based
    # checks below only run when no hand gave a specific match (which is
    # also normal for real crossed arms, since that pose usually hides the
    # hands from the hand model entirely).
    for landmarks in hand_landmarks_list:
        hand_c = landmarks_center(landmarks)

        # Pinch (thumb tip touching index tip) near the face reads as
        # putting on glasses. Checked first and independently of
        # fingers_up/classify_single_hand, since a pinch's finger-extension
        # reading is ambiguous (index is bent to meet the thumb, not
        # cleanly "up" or "down") and would otherwise misclassify.
        if is_pinch(landmarks):
            near_face = np.linalg.norm(hand_c - head_center) < GLASSES_NEAR_FACE_DIST
            if near_face:
                return "glasses", yaw_deg
            continue

        fingers = fingers_up(landmarks)
        gesture = classify_single_hand(fingers)

        # A curled hand (whether the noisy thumb reading calls it "fist" or
        # "thumbs_up") held beside the head is the lollipop gesture. Deciding
        # by position first — instead of trusting the thumb reading — avoids
        # thumb-curl noise turning a beside-head fist into a thumbs-up miss,
        # and keeps a genuine away-from-head thumbs-up from getting diluted
        # by those same noisy votes.
        if gesture in ("fist", "thumbs_up"):
            # Tight zone roughly at cheek height, just off to the side of
            # the face — not the whole upper body. The old thresholds
            # (y<0.28, 0.10<x<0.55) covered more than half the frame, so an
            # ordinary chest-height thumbs-up was still "beside head".
            beside_head = (
                abs(hand_c[1] - head_center[1]) < 0.15
                and 0.08 < abs(hand_c[0] - head_center[0]) < 0.30
            )
            if beside_head:
                return "fist_by_head", yaw_deg
            if gesture == "thumbs_up":
                return ("thumbs_down" if thumb_points_down(landmarks) else "thumbs_up"), yaw_deg
            continue

        if gesture == "pointer":
            # Use the index fingertip (8), not the whole-hand centroid, for
            # distance-to-mouth: averaging all 21 landmarks pulls the
            # "center" back toward the palm/wrist and curled fingers, which
            # sits noticeably farther from the mouth than the actual
            # fingertip does — this was the main reason finger_mouth kept
            # missing even when the finger was right at the mouth.
            fingertip = np.array([landmarks[8].x, landmarks[8].y])
            near_mouth = mouth_point is not None and np.linalg.norm(fingertip - mouth_point) < MOUTH_NEAR_DIST
            if near_mouth:
                return "finger_mouth", yaw_deg
            return "nerd", yaw_deg

    # Pose-based crossed-arms/bicep checked before the "two hands visible"
    # catch-all below — crossed arms often still shows a sliver of both
    # hands to the hand model, which used to win the truck result before
    # the more specific pose-based crossed-arms check ever got a turn.
    if detect_cross_arms(pose_landmarks):
        return "cross_arms", yaw_deg
    if detect_bicep(pose_landmarks):
        return "bicep", yaw_deg

    if len(hand_landmarks_list) == 2:
        return "two_hands", yaw_deg

    if yaw_deg is not None and abs(yaw_deg) > YAW_THRESHOLD_DEG:
        return "side_eye", yaw_deg

    return "default", yaw_deg


def draw_hand_landmarks(square_frame, hand_landmarks_list, orig_w, orig_h, crop_x0, crop_y0):
    # Landmarks are normalized against the original (uncropped) frame, so
    # convert to original-frame pixels first, then shift by the crop
    # origin to land correctly on the square, already-cropped frame.
    for landmarks in hand_landmarks_list:
        points = [
            (int(p.x * orig_w) - crop_x0, int(p.y * orig_h) - crop_y0)
            for p in landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            cv2.line(square_frame, points[a], points[b], (0, 255, 0), 2)
        for x, y in points:
            cv2.circle(square_frame, (x, y), 3, (0, 0, 255), -1)


def main():
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_tracking_confidence=0.6,
        output_facial_transformation_matrixes=True,
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    stable_gesture = "default"
    votes = deque(maxlen=VOTE_WINDOW)
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = frame_index * 33
        frame_index += 1
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        detected, yaw_deg = classify_gesture(hand_result, face_result, pose_result)
        votes.append(detected)

        counts = Counter(votes)
        top_gesture, top_count = counts.most_common(1)[0]
        if top_count >= VOTE_MAJORITY:
            stable_gesture = top_gesture

        # Crop to square BEFORE drawing anything overlaid, then remap
        # landmark coordinates into that cropped frame. Drawing on the full
        # (non-square) frame first — like the old code did — meant the crop
        # step afterward sliced off whatever fell outside the centered
        # square, which for a widescreen webcam is most of the left/right
        # edges, including the fixed (10, 25) corner all the debug text and
        # the hand skeleton lines were drawn at.
        orig_h, orig_w = frame.shape[:2]
        side = min(orig_h, orig_w)
        crop_x0 = (orig_w - side) // 2
        crop_y0 = (orig_h - side) // 2
        square_frame = frame[crop_y0:crop_y0 + side, crop_x0:crop_x0 + side]

        if hand_result.hand_landmarks:
            draw_hand_landmarks(square_frame, hand_result.hand_landmarks, orig_w, orig_h, crop_x0, crop_y0)

        yaw_text = f"{yaw_deg:.1f}" if yaw_deg is not None else "n/a"
        cv2.putText(
            square_frame, f"{detected}  yaw={yaw_text}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
        )
        if hand_result.hand_landmarks:
            first_hand = hand_result.hand_landmarks[0]
            fingers_text = "fingers T,I,M,R,P=" + str(fingers_up(first_hand))
            cv2.putText(
                square_frame, fingers_text, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )
            pinch_text = f"pinch={is_pinch(first_hand)} ({pinch_dist_ratio(first_hand):.2f})  thumb_dy={thumb_dy_ratio(first_hand):.2f}"
            if face_result.face_landmarks:
                face_c = landmarks_center(face_result.face_landmarks[0])
                hand_to_face = np.linalg.norm(landmarks_center(first_hand) - face_c)
                pinch_text += f"  face_dist={hand_to_face:.2f}"
            cv2.putText(
                square_frame, pinch_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )
            if face_result.face_landmarks:
                mouth_lm = face_result.face_landmarks[0][MOUTH_LANDMARK]
                mouth_pt = np.array([mouth_lm.x, mouth_lm.y])
                fingertip_pt = np.array([first_hand[8].x, first_hand[8].y])
                mouth_dist = np.linalg.norm(fingertip_pt - mouth_pt)
                pointer_shape = classify_single_hand(fingers_up(first_hand)) == "pointer"
                cv2.putText(
                    square_frame,
                    f"pointer_shape={pointer_shape}  mouth_dist={mouth_dist:.2f}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                )
        pose_landmarks = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
        if pose_landmarks:
            signals = bicep_signals(pose_landmarks)
            if signals is not None:
                angle, wrist_above, elbow_out = signals
                bicep_text = f"bicep angle={angle:.0f} wrist_above={wrist_above:.2f} elbow_out={elbow_out:.2f}"
            else:
                bicep_text = "bicep: arm not clearly visible"
            cv2.putText(
                square_frame,
                f"cross_arms={detect_cross_arms(pose_landmarks)}  {bicep_text}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
            )

        cam_view = cv2.resize(square_frame, (WINDOW_W, WINDOW_H))
        meme_view = meme_images[stable_gesture]

        combined = np.hstack((meme_view, cam_view))
        cv2.imshow("Gesture Meme", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    hand_landmarker.close()
    face_landmarker.close()
    pose_landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
