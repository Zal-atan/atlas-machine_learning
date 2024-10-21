-- creates a trigger that resets the attribute valid_email only when the email has been changed.
DELIMITER //
DROP TRIGGER IF EXISTS valid_email;

CREATE TRIGGER valid_email
BEFORE UPDATE ON users
FOR EACH ROW
    IF old.email = new.email AND old.valid_email = 0 THEN
        SET new.valid_email = 0;
    ELSEIF old.email = new.email AND old.valid_email = 1 THEN
        SET new.valid_email = 1;
    ELSEIF old.email != new.email THEN
        SET new.valid_email = 0;
    END IF; //

DELIMITER ;

