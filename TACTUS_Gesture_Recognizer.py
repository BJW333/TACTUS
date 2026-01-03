"""
Gesture Control Framework (MediaPipe + OpenCV)

High-level flow:
1) Capture webcam frames (OpenCV).
2) Run MediaPipe HandLandmarker in VIDEO mode with monotonic timestamps.
3) Convert raw landmarks into normalized 21x3 arrays per hand.
4) Detect gestures using:
   - Two-hand gestures first (TWO_PINCH, ROTATE)
   - Single-hand gestures next (SWIPE, PINCH/ROTATE, POINT, SPREAD, GRAB)
5) Smooth gesture output across frames to reduce flicker.
6) Return GestureState + raw MediaPipe results for visualization/debug.

Coordinate conventions:
- Landmarks are normalized in [0..1] for x/y (relative to image width/height).
- Larger x moves right, larger y moves down (image coordinates).
- z is relative depth from MediaPipe (not true metric depth).
"""

import cv2
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable
import time 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
import mediapipe as mp

script_dir = Path(__file__).parent
    
# ──────────────────────────────────────────────────────────────
# Helper Functions (camera setup + debug drawing)
# These functions do NOT affect gesture detection logic.
# ──────────────────────────────────────────────────────────────
def setup_camera():  
    """
    Open default webcam (index 0) and return a cv2.VideoCapture.

    Returns:
        cv2.VideoCapture or None if the camera cannot be opened.
    """  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None
    return cap

def draw_gesture_status(frame, state):
    """
    Draw current gesture name + confidence bar on the frame.

    Visualization details:
    - Text: gesture enum name (e.g., PINCH / ROTATE / NONE)
    - Bar: confidence in [0..1] mapped to 0..200px width

    Note: purely for debugging/UI; does not influence gesture logic.
    """    
    #Gesture name
    text = f"{state.gesture.name}"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
    
    #Confidence bar
    bar_width = int(state.confidence * 200)
    cv2.rectangle(frame, (20, 70), (20 + bar_width, 90), (0, 255, 0), -1)
    cv2.rectangle(frame, (20, 70), (220, 90), (255, 255, 255), 2)
    
    return frame

def draw_landmarks(frame, results):
    """
    Draw a simplified hand skeleton using MediaPipe hand landmarks.

    Input:
        results.hand_landmarks: list of 21 landmarks per detected hand.

    Notes:
    - Coordinates are normalized, so we multiply by (w, h) to draw in pixels.
    - We use a custom connection list rather than mp.solutions drawing utils
      to keep full control over thickness/colors.
    """
    if not results.hand_landmarks:
        return frame
    
    h, w = frame.shape[:2]
    
    #Connection pairs for skeleton
    connections = [
        (0,1),(1,2),(2,3),(3,4),        # Thumb
        (0,5),(5,6),(6,7),(7,8),        # Index
        (0,9),(9,10),(10,11),(11,12),   # Middle
        (0,13),(13,14),(14,15),(15,16), # Ring
        (0,17),(17,18),(18,19),(19,20), # Pinky
        (5,9),(9,13),(13,17)            # Palm
    ]
    
    for hand_landmarks in results.hand_landmarks:
        #Draw connections
        for start_idx, end_idx in connections:
            start = hand_landmarks[start_idx]
            end = hand_landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        #Draw points
        for lm in hand_landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
    
    return frame


#Gesture vocabulary for this system.
#Important: detection order matters (two-hand gestures checked first).
#Some gestures represent "modes" (PINCH/ROTATE), while SWIPE is a discrete action.
class GestureType(Enum):
    NONE = auto()
    PINCH = auto()        #Thumb + index close → Select/Hold (precursor to ROTATE)
    GRAB = auto()         #Closed fist → Move/Drag
    POINT = auto()        #Index extended → Select/Create
    SPREAD = auto()       #All fingers spread → Expand view
    ROTATE = auto()       #Single-hand pinch+drag OR two-hand rotation → Rotate object
    SWIPE_LEFT = auto()   #Quick left motion → Undo
    SWIPE_RIGHT = auto()  #Quick right motion → Redo
    TWO_PINCH = auto()    #Both hands pinching → Precise scale

# HandLandmarks wraps MediaPipe landmarks and provides helper metrics:
# - Common points (wrist, tips)
# - palm_center (average of key palm joints)
# - finger_distance() for pinch/grip thresholds
# - is_finger_extended() heuristic for pose classification
@dataclass
class HandLandmarks:
    """Processed hand landmark data with computed metrics."""
    landmarks: np.ndarray  #21x3 array of landmark positions
    handedness: str        #Left Hand or Right Hand
    
    def __post_init__(self):
        self._palm_center = np.mean(self.landmarks[[0, 5, 9, 13, 17]], axis=0)
        
    @property
    def wrist(self) -> np.ndarray:
        return self.landmarks[0]
    
    @property
    def thumb_tip(self) -> np.ndarray:
        return self.landmarks[4]
    
    @property
    def index_tip(self) -> np.ndarray:
        return self.landmarks[8]
    
    @property
    def middle_tip(self) -> np.ndarray:
        return self.landmarks[12]
    
    @property
    def ring_tip(self) -> np.ndarray:
        return self.landmarks[16]
    
    @property
    def pinky_tip(self) -> np.ndarray:
        return self.landmarks[20]
    
    # palm_center is used as the "hand position" for motion/velocity tracking.
    # It's more stable than fingertip points, which jitter more.
    @property
    def palm_center(self) -> np.ndarray:
        return self._palm_center
    
    def finger_distance(self, finger1_idx: int, finger2_idx: int) -> float:
        a = self.landmarks[finger1_idx][:2]  # XY only
        b = self.landmarks[finger2_idx][:2]
        return float(np.linalg.norm(a - b))
    
    # Extension heuristic:
    # If fingertip is significantly farther from wrist than MCP joint,
    # treat the finger as extended. Multiplier tunes strictness.
    def is_finger_extended(self, finger_tip_idx: int, finger_mcp_idx: int) -> bool:
        tip = self.landmarks[finger_tip_idx][:2]
        mcp = self.landmarks[finger_mcp_idx][:2]
        wrist = self.wrist[:2]
        return np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.15

# GestureState is the per-frame output of the recognizer.
# It carries:
# - gesture: smoothed final gesture label
# - confidence: current confidence (0..1)
# - position: current hand position (normalized)
# - velocity: smoothed velocity estimate (normalized units/sec)
# - scale_factor: used only for TWO_PINCH
# - rotation_delta: used only for ROTATE
# - timestamp: monotonic seconds from time.monotonic()
@dataclass
class GestureState:
    """Current gesture state with confidence and momentum."""
    gesture: GestureType = GestureType.NONE
    confidence: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale_factor: float = 1.0
    rotation_delta: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0
    

class Gesture_Recognizer:
    """
    Gesture_Recognizer processes video frames to detect hand gestures.

    Core responsibilities:
    - Run MediaPipe HandLandmarker on each frame.
    - Maintain short-term temporal state (previous positions, previous gesture).
    - Detect gestures based on:
    (A) Two-hand rules (higher priority)
    (B) Single-hand rules (lower priority)
    - Smooth the gesture label to prevent flicker.
    - Provide optional callback hooks on gesture changes.

    Design note:
    - All thresholds are in normalized landmark coordinates (not pixels).
    - Small threshold tweaks can have large UX impact depending on camera distance.
    """
    # ──────────────────────────────────────────────────────────
    # Tuning Constants (normalized units)
    #
    # PINCH_THRESHOLD: thumb-index distance to trigger pinch mode.
    # ROTATE_GRIP_THRESHOLD: looser pinch threshold used to keep rotate active.
    # ROTATE_MOVE_THRESHOLD: minimum per-frame movement before we call it ROTATE.
    # ROTATE_GAIN: scales how strongly motion maps to rotation_delta.
    #
    # TWO_HAND_ROTATE_MIN: min angle delta between hands to trigger 2-hand rotate.
    #
    # Swipe parameters:
    # SWIPE_VEL_START: x-velocity required to begin swipe tracking.
    # SWIPE_MIN_DISPLACEMENT: total x displacement required to confirm swipe.
    # SWIPE_MAX_TIME: must complete within this time window.
    # SWIPE_COOLDOWN: ignore new swipes briefly after a swipe.
    # ──────────────────────────────────────────────────────────
    
    #pinch / grip (normalized)
    PINCH_THRESHOLD = 0.055          # was 0.08 (too loose + distance-sensitive)
    ROTATE_GRIP_THRESHOLD = 0.095    # was 0.15 (too loose)

    # movement threshold for rotate gesture
    ROTATE_MOVE_THRESHOLD = 0.008   # try 0.006–0.010
    # rotation gain for rotate gesture
    ROTATE_GAIN = 1.4  # try 1.6–2.4
    
    #motion
    VELOCITY_SMOOTHING = 0.25        # slightly snappier than 0.3

    #two-hand rotate tuning
    TWO_HAND_ROTATE_MIN = 0.09  # try 0.08–0.12

    #swipe tuning (normalized units per second)
    SWIPE_VEL_START        = 0.55  # was 0.75 (easier to start)
    SWIPE_MIN_DISPLACEMENT = 0.07  # was 0.10 (less distance needed)
    SWIPE_MAX_TIME         = 0.70  # was 0.60 (more time to complete)
    SWIPE_COOLDOWN         = 0.30  # keep same

    SWIPE_HORIZ_DOMINANCE  = 1.3   # was 1.6 (more forgiving angle)
    SWIPE_MAX_VERT_DRIFT   = 0.09  # was 0.06 (allow more wobble)
    
    def __init__(self, model_path: str | Path = script_dir / "hand_landmarker.task"):
        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        # MediaPipe Tasks model setup:
        # - running_mode=VIDEO requires increasing timestamps in milliseconds
        # - num_hands=2 enables 2-hand gestures
        # - confidences balance detection sensitivity vs jitter
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        
        # Previous frame state used for:
        # - velocity estimation
        # - "was_gripping" logic (pinch/rotate continuity)
        # - detecting gesture transitions for callbacks
        self.prev_state = GestureState()
        self.gesture_callbacks: dict[GestureType, list[Callable]] = {g: [] for g in GestureType}
        self.frame_count = 0

        #For rotation gesture
        self.prev_two_hand_angle = None #For two-hand rotation

        # Per-hand history:
        # prev_hand[side] stores (previous palm_center, previous timestamp)
        # so we can compute per-frame deltas and velocity-like values.
        self.prev_hand = {"Left": None, "Right": None}       # (pos, ts)

        # Gesture smoothing:
        # Requires the same gesture to appear for N frames before "locking" it in.
        # Exception: swipe + two-pinch are treated as instant actions/modes.
        self.gesture_buffer = []
        self.GESTURE_FRAMES_REQUIRED = 3  #Must see same gesture for 3 frames
        self.confirmed_gesture = GestureType.NONE
        
        self.start_time = None
        self.last_timestamp_ms = 0
        
        self.prev_two_pinch_dist = None
        
        # Swipe state machine:
        # active: currently tracking a potential swipe
        # start_x/start_t: start point/time of candidate swipe
        # dir: +1 for right swipe, -1 for left swipe
        # swipe_cooldown_until prevents repeated accidental triggers
        self.swipe_tracker = {
            "Left":  {"active": False, "start_x": 0.0, "start_y": 0.0, "start_t": 0.0, "dir": 0},
            "Right": {"active": False, "start_x": 0.0, "start_y": 0.0, "start_t": 0.0, "dir": 0},
        }
        self.swipe_cooldown_until = 0.0
        
    def register_callback(self, gesture: GestureType, callback: Callable):
        """Register a callback for when a gesture is detected."""
        self.gesture_callbacks[gesture].append(callback)
    
    def _invoke_callbacks(self, gesture: GestureType, state: GestureState):
        """Invoke all registered callbacks for a detected gesture."""
        for callback in self.gesture_callbacks[gesture]:
            callback(state)
            
    def _reset_no_hands(self, current_state):
        self.prev_hand["Left"] = None
        self.prev_hand["Right"] = None
        self.prev_two_pinch_dist = None
        self.prev_two_hand_angle = None

        self.gesture_buffer.clear()
        self.confirmed_gesture = GestureType.NONE

        #fully reset swipe state
        for k in ("Left", "Right"):
            self.swipe_tracker[k] = {"active": False, "start_x": 0.0, "start_y": 0.0, "start_t": 0.0, "dir": 0}
            
        self.prev_state = current_state
    
    def _smooth_gesture(self, detected: GestureType) -> GestureType:
        """Only change gesture after N consistent frames."""
        
        # Bypass smoothing for these 
        if detected == GestureType.NONE:
            self.gesture_buffer.clear()
            self.confirmed_gesture = GestureType.NONE
            return GestureType.NONE
        
        if detected in (GestureType.SWIPE_LEFT, GestureType.SWIPE_RIGHT, GestureType.TWO_PINCH):
            self.gesture_buffer.clear()
            self.confirmed_gesture = detected
            return detected
        
        # Normal smoothing for other gestures
        self.gesture_buffer.append(detected)
        
        if len(self.gesture_buffer) > self.GESTURE_FRAMES_REQUIRED:
            self.gesture_buffer.pop(0)
        
        if len(self.gesture_buffer) == self.GESTURE_FRAMES_REQUIRED:
            if all(g == self.gesture_buffer[0] for g in self.gesture_buffer):
                self.confirmed_gesture = self.gesture_buffer[0]
        
        return self.confirmed_gesture
    
    def _sort_hands(self, hands_data):
        # Ensure stable ordering: Left first, then Right
        order = {"Left": 0, "Right": 1}
        return sorted(hands_data, key=lambda h: order.get(h.handedness, 99))  
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> tuple:
        """
        Process one frame:
        1) Convert BGR -> RGB and wrap as mp.Image
        2) Create monotonic timestamp in ms for MediaPipe VIDEO mode
        3) Run HandLandmarker.detect_for_video
        4) Convert raw landmarks into HandLandmarks objects
        5) Detect gestures (two-hand first, then single-hand)
        6) Compute velocity (smoothed)
        7) Smooth gesture label (temporal filtering)
        8) Fire callbacks when gesture changes
        9) Update previous frame caches and return (state, results)
        """
        #Convert to MediaPipe Image format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        #Timestamp must be monotonically increasing milliseconds
        self.frame_count += 1
        
        if self.start_time is None:
            self.start_time = timestamp

        # MediaPipe VIDEO mode requires monotonically increasing timestamps (ms).
        # We compute ms from first frame, then enforce strict monotonicity by
        # bumping by +1 if the computed value repeats or goes backward.
        computed_ms = int((timestamp - self.start_time) * 1000)
        timestamp_ms = max(self.last_timestamp_ms + 1, computed_ms)  # enforce monotonic
        self.last_timestamp_ms = timestamp_ms
        
        #Perform hand landmark detection
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        current_state = GestureState(timestamp=timestamp)
        
        # If no hands detected:
        # - Clear all per-hand history to avoid stale deltas
        # - Reset two-hand gesture memory (pinch dist, angle)
        # - Reset gesture smoothing buffer
        # - Reset swipe tracking state
        # Then return immediately.
        if not results.hand_landmarks:
            self._reset_no_hands(current_state)
            return current_state, results

        #If hands exist: build hands_data
        # Convert MediaPipe output into a stable list of HandLandmarks objects.
        # We also attach handedness ("Left"/"Right") so gestures can maintain
        # consistent per-hand state across frames.
        hands_data = []
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
            if results.handedness and len(results.handedness) > i and results.handedness[i]:
                raw_handedness = results.handedness[i][0].category_name  # Get from MediaPipe first
                handedness = "Right" if raw_handedness == "Left" else "Left"  # Then invert
            else:
                continue  # skip unknown handedness safely

            if handedness not in ("Left", "Right"):
                continue # skip unknown handedness safely

            hands_data.append(HandLandmarks(landmarks=landmarks, handedness=handedness))
            
        if not hands_data:
            self._reset_no_hands(current_state)
            return current_state, results
            
        # Sort hands so Left/Right ordering is stable frame-to-frame.
        # Snapshot prev_hand BEFORE updates so deltas use the last frame.
        hands_data = self._sort_hands(hands_data)
        
        # snapshot previous positions BEFORE we update anything
        prev_snapshot = dict(self.prev_hand)

        # If one hand disappears this frame, clear its cached history.
        # This prevents stale previous positions from causing velocity spikes
        # or swipe false-positives when the hand re-enters the frame.
        seen = {h.handedness for h in hands_data}
        for side in ("Left", "Right"):
            if side not in seen:
                self.prev_hand[side] = None
                self.swipe_tracker[side] = {"active": False, "start_x": 0.0, "start_y": 0.0, "start_t": 0.0, "dir": 0}
                
        # ═══════════════════════════════════════════════════════════
        # TWO-HAND GESTURES (check first)
        # ═══════════════════════════════════════════════════════════
        # Two-hand gestures get priority over single-hand gestures.
        # If a two-hand gesture is recognized, we skip single-hand detection.
        if len(hands_data) == 2 and {hands_data[0].handedness, hands_data[1].handedness} == {"Left", "Right"}:
            hand1 = hands_data[0] 
            hand2 = hands_data[1]

            hand1_pinching = hand1.finger_distance(4, 8) < self.PINCH_THRESHOLD
            hand2_pinching = hand2.finger_distance(4, 8) < self.PINCH_THRESHOLD

            if hand1_pinching and hand2_pinching:
                # TWO_PINCH: both hands pinching -> treat as scale gesture.
                # We compute scale factor from ratio of current hand distance
                # to previous hand distance.
                hand_dist = float(np.linalg.norm(hand1.palm_center - hand2.palm_center))

                if self.prev_two_pinch_dist is None:
                    scale = 1.0
                else:
                    scale = hand_dist / max(self.prev_two_pinch_dist, 1e-6)

                self.prev_two_pinch_dist = hand_dist
                self.prev_two_hand_angle = None  # prevents rotate “jump” after pinch

                current_state.gesture = GestureType.TWO_PINCH
                current_state.confidence = 1.0
                current_state.scale_factor = scale
                current_state.position = (hand1.palm_center + hand2.palm_center) / 2

            else:
                # not pinching -> clear pinch state
                self.prev_two_pinch_dist = None

                # ROTATE requires both hands SPREAD
                hand1_spread = all(hand1.is_finger_extended(tip, mcp) for tip, mcp in [(8,5),(12,9),(16,13),(20,17)])
                hand2_spread = all(hand2.is_finger_extended(tip, mcp) for tip, mcp in [(8,5),(12,9),(16,13),(20,17)])

                if hand1_spread and hand2_spread:
                    # Two-hand ROTATE:
                    # Use the angle of the vector between palm centers.
                    # delta angle between frames becomes rotation_delta (z-axis).
                    angle = np.arctan2(
                        hand2.palm_center[1] - hand1.palm_center[1],
                        hand2.palm_center[0] - hand1.palm_center[0]
                    )

                    if self.prev_two_hand_angle is not None:
                        angle_delta = angle - self.prev_two_hand_angle
                        if angle_delta > np.pi:
                            angle_delta -= 2*np.pi
                        elif angle_delta < -np.pi:
                            angle_delta += 2*np.pi

                        if abs(angle_delta) > self.TWO_HAND_ROTATE_MIN:
                            current_state.gesture = GestureType.ROTATE
                            current_state.confidence = min(abs(angle_delta) / 0.15, 1.0)
                            current_state.rotation_delta = np.array([0, 0, angle_delta])
                            current_state.position = (hand1.palm_center + hand2.palm_center) / 2

                    self.prev_two_hand_angle = angle
                else:
                    self.prev_two_hand_angle = None

        else:
            # not two hands -> clear ONLY two-hand gesture state
            self.prev_two_pinch_dist = None
            self.prev_two_hand_angle = None
                
        # ═══════════════════════════════════════════════════════════
        # SINGLE-HAND GESTURES (only if no two-hand gesture)
        # ═══════════════════════════════════════════════════════════
        if current_state.gesture == GestureType.NONE:
            # Single-hand gestures run only when we did NOT detect a two-hand gesture.
            # Order matters:
            # - SWIPE first (discrete action)
            # - PINCH/ROTATE next (mode-based)
            # - POINT / SPREAD / GRAB pose-based 
            for hand in hands_data:
                
                # ──────────────────────────────────────────────────
                # SWIPE (robust): displacement within time window
                # ──────────────────────────────────────────────────
                # SWIPE detection:
                # 1) look at x-velocity from previous frame
                # 2) if it exceeds threshold, begin tracking candidate swipe
                # 3) confirm swipe if total displacement crosses threshold within time window
                #
                # Note: overly-sensitive swipe usually comes from:
                # - low SWIPE_VEL_START (arms too easily)
                # - low SWIPE_MIN_DISPLACEMENT (confirms too easily)
                # - lack of horizontal/pose gating (e.g., triggers during rotate motion)
                key = hand.handedness
                if key not in self.swipe_tracker:
                    continue
                tr = self.swipe_tracker[key]   
                prev = prev_snapshot.get(key)

                if prev is None:
                    tr["active"] = False
                    
                #Per-hand delta (prevents rotate jumpiness)
                delta_x = 0.0
                delta_y = 0.0
                if prev is not None:
                    prev_pos, prev_ts = prev
                    delta_x = float(hand.palm_center[0] - prev_pos[0])
                    delta_y = float(hand.palm_center[1] - prev_pos[1])
                
                if timestamp >= self.swipe_cooldown_until and prev is not None:
                    prev_pos, prev_ts = prev
                    dt = timestamp - prev_ts
                    if dt > 1e-6:  # avoid near-zero division
                        dx = float(hand.palm_center[0] - prev_pos[0])
                        dy = float(hand.palm_center[1] - prev_pos[1])
                        vel_x = dx / dt
                        vel_y = dy / dt

                        # start tracking swipe ONLY if horizontal dominates vertical
                        if (not tr["active"]) and abs(vel_x) > self.SWIPE_VEL_START and abs(vel_x) > abs(vel_y) * self.SWIPE_HORIZ_DOMINANCE:
                            tr["active"] = True
                            tr["start_x"] = float(prev_pos[0])
                            tr["start_t"] = float(prev_ts)
                            tr["start_y"] = float(prev_pos[1])   # NEW (vertical drift gate)
                            tr["dir"] = 1 if vel_x > 0 else -1

                        if tr["active"]:
                            elapsed = timestamp - tr["start_t"]
                            total_dx = float(hand.palm_center[0] - tr["start_x"])
                            total_dy = float(hand.palm_center[1] - tr["start_y"])  

                            # cancel if too much vertical drift (diagonal movement)
                            if abs(total_dy) > self.SWIPE_MAX_VERT_DRIFT:
                                tr["active"] = False
                            else:
                                # success
                                if elapsed <= self.SWIPE_MAX_TIME and (tr["dir"] * total_dx) >= self.SWIPE_MIN_DISPLACEMENT:
                                    current_state.gesture = GestureType.SWIPE_RIGHT if tr["dir"] > 0 else GestureType.SWIPE_LEFT
                                    current_state.confidence = 1.0
                                    current_state.position = hand.palm_center

                                    tr["active"] = False
                                    self.swipe_cooldown_until = timestamp + self.SWIPE_COOLDOWN
                                    break

                                # fail: timed out OR net reversed
                                if elapsed > self.SWIPE_MAX_TIME or (tr["dir"] * total_dx) < -0.03:
                                    tr["active"] = False

                #If still in cooldown, deactivate swipe tracking
                if timestamp < self.swipe_cooldown_until:
                    tr["active"] = False
                    
                #If swipe is being tracked, don't allow other gestures to interrupt it
                if tr["active"]:
                    continue

                #PINCH / ROTATE (grip-based):
                # - Both begin from thumb-index proximity ("grip").
                # - ROTATE requires meaningful motion while gripping.
                # - If gripping but mostly stationary, remain in PINCH.
                #
                # The "was_gripping" check prevents mode flicker across frames.
                thumb_index_dist = hand.finger_distance(4, 8)
                is_tight_pinch = thumb_index_dist < self.PINCH_THRESHOLD       
                is_loose_grip = thumb_index_dist < self.ROTATE_GRIP_THRESHOLD 
                
                # Make sure other fingers aren't all curled (that's GRAB, not PINCH)
                other_fingers_curled = not any(
                    hand.is_finger_extended(tip, mcp)
                    for tip, mcp in [(12, 9), (16, 13), (20, 17)]
                )

                if is_tight_pinch and not other_fingers_curled:
                    was_gripping = self.prev_state.gesture in (GestureType.PINCH, GestureType.ROTATE)
                    
                    if was_gripping and prev is not None:
                        # Only ROTATE if movement exceeds threshold
                        if abs(delta_x) > self.ROTATE_MOVE_THRESHOLD or abs(delta_y) > self.ROTATE_MOVE_THRESHOLD:
                            current_state.gesture = GestureType.ROTATE
                            current_state.rotation_delta = np.array([delta_y * self.ROTATE_GAIN, delta_x * self.ROTATE_GAIN, 0])
                        else:
                            current_state.gesture = GestureType.PINCH
                    else:
                        current_state.gesture = GestureType.PINCH
                    
                    current_state.confidence = max(0.0, 1.0 - (thumb_index_dist / self.PINCH_THRESHOLD))
                    current_state.position = hand.palm_center
                    break
                
                if is_loose_grip and not other_fingers_curled:
                    was_gripping = self.prev_state.gesture in (GestureType.PINCH, GestureType.ROTATE)
                    
                    if was_gripping and prev is not None:
                        # Only ROTATE if movement exceeds threshold
                        if abs(delta_x) > self.ROTATE_MOVE_THRESHOLD or abs(delta_y) > self.ROTATE_MOVE_THRESHOLD:
                            current_state.gesture = GestureType.ROTATE
                            # Apply gain to make rotation more responsive
                            current_state.rotation_delta = np.array([ 
                                delta_y * self.ROTATE_GAIN,
                                delta_x * self.ROTATE_GAIN,
                                0
                            ])
                        else:
                            current_state.gesture = GestureType.PINCH  # Holding still
                    else:
                        current_state.gesture = GestureType.PINCH  # First frame of grip
                    
                    current_state.confidence = max(0.0, 1.0 - (thumb_index_dist / self.ROTATE_GRIP_THRESHOLD))
                    current_state.position = hand.palm_center
                    break
                
                # NOTE (future): "NEUTRAL" pose
                # Consider adding a neutral/open-relaxed gesture that:
                # - does NOT trigger actions
                # - can be used as an explicit "idle" state
                # This helps avoid accidental gestures when the user is resting.
                
                # ──────────────────────────────────────────────────
                # Pose-based gestures:
                # - POINT: index extended, other fingers curled
                # - SPREAD: all main fingers extended
                # - GRAB: all main fingers curled (fist-like)
                # ──────────────────────────────────────────────────
                
                #POINT
                index_extended = hand.is_finger_extended(8, 5)
                others_curled = not any(
                    hand.is_finger_extended(tip, mcp) 
                    for tip, mcp in [(12, 9), (16, 13), (20, 17)]
                )
                if index_extended and others_curled:
                    current_state.gesture = GestureType.POINT
                    current_state.confidence = 1.0
                    current_state.position = hand.palm_center
                    break
                
                #SPREAD
                all_extended = all(
                    hand.is_finger_extended(tip, mcp)
                    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]
                )
                if all_extended:
                    current_state.gesture = GestureType.SPREAD
                    current_state.confidence = 1.0
                    current_state.position = hand.palm_center
                    break
                
                #GRAB
                all_curled = not any(
                    hand.is_finger_extended(tip, mcp)
                    for tip, mcp in [(8, 5), (12, 9), (16, 13), (20, 17)]  # Fingers only (thumb unreliable)
                )
                if all_curled:
                    current_state.gesture = GestureType.GRAB
                    current_state.confidence = 1.0
                    current_state.position = hand.palm_center
                    break
        
        #Set position from first hand if not set
        if np.allclose(current_state.position, 0.0) and hands_data:
            current_state.position = hands_data[0].palm_center
    
        # ═══════════════════════════════════════════════════════════
        # VELOCITY CALCULATION
        # ═══════════════════════════════════════════════════════════
        # Velocity estimation:
        # We compute instantaneous velocity from position delta / time delta.
        # Then apply exponential smoothing (VELOCITY_SMOOTHING) so velocity
        # doesn’t jitter wildly frame-to-frame.
        prev_pos = self.prev_state.position
        dt = timestamp - self.prev_state.timestamp
        if dt > 1e-6 and not np.allclose(prev_pos, 0.0):  # avoid near-zero division
            instant_velocity = (current_state.position - prev_pos) / dt
            current_state.velocity = (
                self.VELOCITY_SMOOTHING * instant_velocity +
                (1 - self.VELOCITY_SMOOTHING) * self.prev_state.velocity
            )
        else:
            current_state.velocity = np.zeros(3)
            
        # ═══════════════════════════════════════════════════════════
        # GESTURE SMOOTHING
        # ═══════════════════════════════════════════════════════════
        raw_gesture = current_state.gesture
        # Gesture smoothing:
        # Convert raw detected gesture into a smoothed label that only changes
        # after N consistent frames (except for discrete gestures like swipe).
        smoothed = self._smooth_gesture(raw_gesture)
        current_state.gesture = smoothed

        #If smoothing changed the gesture, clear gesture-specific payloads
        if smoothed != raw_gesture:
            if smoothed != GestureType.ROTATE:
                current_state.rotation_delta = np.zeros(3)
            if smoothed != GestureType.TWO_PINCH:
                current_state.scale_factor = 1.0
            # keep confidence as-is
            
        #Invoke callbacks on gesture change
        if current_state.gesture != self.prev_state.gesture:
            self._invoke_callbacks(current_state.gesture, current_state)
        
        # Update per-hand cached positions AFTER all detection logic.
        # This ensures deltas computed above always compare against last frame.
        for h in hands_data:
            self.prev_hand[h.handedness] = (h.palm_center.copy(), timestamp)
            
        self.prev_state = current_state
        return current_state, results 
        
    
def main():
    """
    Demo runner:
    - Opens camera
    - Runs recognizer per-frame
    - Draws landmarks + gesture status overlay
    - Prints detected gesture/confidence to console
    - Press 'q' to exit
    """
    print("Starting...")
    cap = setup_camera()
    if cap is None:
        print("Camera failed to open")
        return
    
    print("Camera opened successfully")
    recognizer = Gesture_Recognizer()
    print("Recognizer created, entering loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        timestamp = time.monotonic()
        
        state, results = recognizer.process_frame(frame, timestamp)
        
        #Draw landmarks on the frame 
        frame = draw_landmarks(frame, results)
        frame = draw_gesture_status(frame, state)  

        if state.gesture != GestureType.NONE:
            print(f"{state.gesture.name}: {state.confidence:.0%}")
        
        cv2.imshow("Gesture Control Framework", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()