CREATE TABLE migration_runner_case_expression (
    value boolean PRIMARY KEY,
    result text NOT NULL
);

INSERT INTO migration_runner_case_expression (value, result)
VALUES (
    true,
    CASE
        WHEN true THEN 'operator'
        ELSE 'none'
    END
);
