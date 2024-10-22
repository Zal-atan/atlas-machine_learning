-- creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER //

DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)
BEGIN
    -- Update the user's average score directly with a single query
    UPDATE users
    SET average_score = (
        SELECT IFNULL(AVG(score), 0) -- Set average score to 0 if no scores exist
        FROM corrections
        WHERE corrections.user_id = user_id
    )
    WHERE users.id = user_id;
END; //

DELIMITER ;
