# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
# 5th Edited by ControlNet (Improved JSON serialization/deserialization, and lots of bug fixs)
# This preprocessor is licensed by CMU for non-commercial use only.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from typing import Tuple, List, Callable, Union, Optional

from . import util
from .wholebody import Wholebody  # DW Pose
from .body import Body, BodyResult, Keypoint
from .hand import Hand
from .face import Face
from .types import HandResult, FaceResult, HumanPoseResult, AnimalPoseResult
from .animal_pose import AnimalPose, draw_animal_poses


def draw_human_poses(
    poses: List[HumanPoseResult], H, W, draw_body=True, draw_face=True, draw_hand=True, 
):
    """
    Draw the detected poses on an empty canvas.

    Args:
        poses (List[HumanPoseResult]): A list of HumanPoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with the drawn poses.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = util.draw_bodypose(canvas, pose.body.keypoints)

        if draw_hand:
            canvas = util.draw_handpose(canvas, pose.left_hand)
            canvas = util.draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = util.draw_facepose(canvas, pose.face)

    return canvas


def decode_json_as_poses(
    pose_json: dict,
) -> Tuple[List[HumanPoseResult], List[AnimalPoseResult], int, int]:
    """
    Decode the json_string complying with the OpenPose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.

    Returns:
        human_poses
        animal_poses
        canvas_height
        canvas_width
    """
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def decompress_keypoints(numbers: Optional[List[float]]) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            HumanPoseResult(
                      body=BodyResult(
                 keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))),
                 left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                      face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json.get("people", [])
        ],
        [decompress_keypoints(pose) for pose in pose_json.get("animals", [])],
        height,
        width,
    )


def encode_poses_as_json(
    poses: List[HumanPoseResult],
    animals: List[AnimalPoseResult],
    canvas_height: int,
    canvas_width: int,
) -> dict:
    """Encode the pose as a JSON compatible dict following openpose JSON output format:
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
    """

    def compress_keypoints(keypoints: Union[List[Keypoint], None]) -> Union[List[float], None]:
        if not keypoints:
            return None

        return [
            value
            for keypoint in keypoints
            for value in (
                [float(keypoint.x), float(keypoint.y), 1.0]
                if keypoint is not None
                else [0.0, 0.0, 0.0]
            )
        ]

    return {
        "people": [
            {
                    "pose_keypoints_2d": compress_keypoints(pose.body.keypoints),
                    "face_keypoints_2d": compress_keypoints(pose.face),
                "hand_left_keypoints_2d": compress_keypoints(pose.left_hand),
                "hand_right_keypoints_2d": compress_keypoints(pose.right_hand),
            }
            for pose in poses
        ],
        "animals": [compress_keypoints(animal) for animal in animals],
        "canvas_height": canvas_height,
        "canvas_width": canvas_width,
    }


def detect_hands(
    image, hand_estimator, body: BodyResult, 
) -> Tuple[Union[HandResult, None], 
           Union[HandResult, None]]:

    H, W, _ = image.shape

    left_hand = None
    right_hand = None

    for x, y, w, is_left in util.handDetect(body, image):
        peaks = hand_estimator(image[y : y + w, 
                                     x : x + w, :]).astype(np.float32)
        if peaks.ndim == 2 and peaks.shape[1] == 2:
            peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
            peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)

            hand_result = [Keypoint(x=peak[0], 
                                    y=peak[1]) for peak in peaks]
            if is_left:
                left_hand = hand_result
            else:
                right_hand = hand_result

    return left_hand, right_hand


def detect_face(image, face_estimator, body: BodyResult) -> Union[FaceResult, None]:

    face = util.faceDetect(body, image)
    if face is None:
        return None

    H, W, _ = image.shape
    x, y, w = face

    heatmaps = face_estimator(image[y : y + w, 
                                    x : x + w, :])
    peaks = face_estimator.compute_peaks_from_heatmaps(heatmaps).astype(np.float32)
    if peaks.ndim == 2 and peaks.shape[1] == 2:
        peaks[:, 0] = np.where(peaks[:, 0] < 1e-6, -1, peaks[:, 0] + x) / float(W)
        peaks[:, 1] = np.where(peaks[:, 1] < 1e-6, -1, peaks[:, 1] + y) / float(H)
        return [Keypoint(x=peak[0], 
                         y=peak[1]) for peak in peaks]
    return None


def detect_human_pose(    
    image, 
    body_estimator, 
    face_estimator=None,
    hand_estimator=None,
    device=None,
) -> List[HumanPoseResult]:
    """
    Detect poses in the given image.
        Args:
            image (numpy.ndarray): The input image for pose detection.

    Returns:
        List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
    """
    include_face = face_estimator is not None
    include_hand = hand_estimator is not None

    if device is not None:
        body_estimator.model.to(device)
        body_estimator.cn_device = device

        if include_hand:
            hand_estimator.model.to(device)
            hand_estimator.cn_device = device

        if include_face:
            face_estimator.model.to(device)
            face_estimator.cn_device = device

    image = image[:, :, ::-1].copy()
    H, W, C = image.shape

    with torch.no_grad():
        candidate, \
        subset = body_estimator(image)
        bodies = body_estimator.format_body_result(candidate, subset)

        results = []
        for body in bodies:
            left_hand, right_hand, face = [None] * 3
            if include_hand:
                left_hand, right_hand = detect_hands(image, hand_estimator, body)
            if include_face:
                face = detect_face(image, face_estimator, body)

            results.append(
                HumanPoseResult(
                    BodyResult(
                        total_score=body.total_score,
                        total_parts=body.total_parts,
                        keypoints=[Keypoint(x = keypoint.x / float(W), 
                                            y = keypoint.y / float(H))
                                if keypoint is not None else None
                                for keypoint in body.keypoints],
                    ),
                    left_hand,
                    right_hand,
                    face,
                )
            )

        return results


def detect_human_pose_dw(image, pose_estimator) -> List[HumanPoseResult]:
    """
    Detect poses in the given image using DW Pose:
        https://github.com/IDEA-Research/DWPose

    Args:
        image (numpy.ndarray): The input image for pose detection.

    Returns:
        List[HumanPoseResult]: A list of HumanPoseResult objects containing the detected poses.
    """
    with torch.no_grad():
        keypoints_info = pose_estimator(image.copy())
        return Wholebody.format_result(keypoints_info)


def detect_animal_pose(image, animal_pose_estimator) -> List[AnimalPoseResult]:
    """
    Detect poses in the given image using RTMPose AP10k model:
        https://github.com/abehonest/ControlNet_AnimalPose

    Args:
        image (numpy.ndarray): The input image for pose detection.

    Returns:
        A list of AnimalPoseResult objects containing the detected animal poses.
    """
    with torch.no_grad():
        return animal_pose_estimator(image.copy())
