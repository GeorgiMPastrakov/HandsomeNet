from handsomenet.constants import JOINT_NAMES, NUM_JOINTS, SKELETON_EDGES


def test_joint_count_matches_constant() -> None:
    assert len(JOINT_NAMES) == NUM_JOINTS == 21


def test_skeleton_has_expected_number_of_edges() -> None:
    assert len(SKELETON_EDGES) == 20

