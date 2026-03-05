def build_grasp_request(object_id: str, grasp_point_id: str, grasp_type: str) -> dict:
    return {
        "object_id": object_id,
        "grasp_point_id": grasp_point_id,
        "grasp_type": grasp_type,
    }
