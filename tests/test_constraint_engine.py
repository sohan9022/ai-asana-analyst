from app.engine.constraint_engine import run_constraint_check, calculate_angle

def test_angle_calculation():
    """Test that angle calculation works correctly."""
    # A straight line should be ~180 degrees
    a = (0, 0)
    b = (1, 0)
    c = (2, 0)
    angle = calculate_angle(a, b, c)
    assert 178 <= angle <= 180, f"Expected ~180, got {angle}"
    print(f"Straight line angle: {angle}° ✅")

def test_warrior_ii_violation():
    """Test that a bad Warrior II front knee is flagged."""
    joint_angles = {
        "front_knee": 120.0,  # Too wide — should be 80-100
        "back_knee":  175.0,  # Fine
    }
    violations = run_constraint_check("Warrior II", joint_angles)
    assert len(violations) > 0, "Should have caught front knee violation"
    assert violations[0]["joint"] == "front_knee"
    print(f"Violation caught: {violations[0]['mistake']} ✅")
    print(f"Correction: {violations[0]['correction']}")

def test_warrior_ii_correct():
    """Test that a correct Warrior II passes with no violations."""
    joint_angles = {
        "front_knee": 90.0,   # Perfect
        "back_knee":  175.0,  # Perfect
    }
    violations = run_constraint_check("Warrior II", joint_angles)
    angle_violations = [v for v in violations if v["type"] == "angle"]
    assert len(angle_violations) == 0, "Should have no angle violations"
    print("Correct Warrior II — no violations ✅")

if __name__ == "__main__":
    test_angle_calculation()
    test_warrior_ii_violation()
    test_warrior_ii_correct()
    print("\nAll tests passed! Constraint Engine is working correctly 🎉")