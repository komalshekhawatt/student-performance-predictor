-- Table: students (load notebook/data/stud.csv; columns renamed with underscores)
-- Columns: gender, race_ethnicity, parental_level_of_education, lunch,
--          test_preparation_course, math_score, reading_score, writing_score

-- 1. Overall KPIs
SELECT COUNT(*) AS total_students,
       ROUND(AVG(math_score),2) AS avg_math,
       ROUND(AVG(reading_score),2) AS avg_reading,
       ROUND(AVG(writing_score),2) AS avg_writing
FROM students;

-- 2. Does the test prep course help?
SELECT test_preparation_course, COUNT(*) AS students,
       ROUND(AVG(math_score),1) AS avg_math,
       ROUND(AVG(reading_score),1) AS avg_reading,
       ROUND(AVG(writing_score),1) AS avg_writing
FROM students GROUP BY test_preparation_course;

-- 3. Lunch type vs math score
SELECT lunch, COUNT(*) AS students, ROUND(AVG(math_score),1) AS avg_math
FROM students GROUP BY lunch ORDER BY avg_math DESC;

-- 4. Parental education vs math score
SELECT parental_level_of_education, COUNT(*) AS students, ROUND(AVG(math_score),1) AS avg_math
FROM students GROUP BY parental_level_of_education ORDER BY avg_math DESC;

-- 5. Gender gap by subject
SELECT gender, ROUND(AVG(math_score),1) AS math,
       ROUND(AVG(reading_score),1) AS reading, ROUND(AVG(writing_score),1) AS writing
FROM students GROUP BY gender;

-- 6. Top 10 students by overall average
SELECT *, ROUND((math_score+reading_score+writing_score)/3.0,1) AS avg_score
FROM students ORDER BY avg_score DESC LIMIT 10;

-- 7. Students at risk (math below 40)
SELECT COUNT(*) AS at_risk,
       ROUND(100.0*COUNT(*)/(SELECT COUNT(*) FROM students),1) AS pct_of_all
FROM students WHERE math_score < 40;

-- 8. At-risk students: which group do they belong to?
SELECT lunch, test_preparation_course, COUNT(*) AS at_risk_students
FROM students WHERE math_score < 40
GROUP BY lunch, test_preparation_course ORDER BY at_risk_students DESC;

-- 9. Combined effect: lunch + test prep
SELECT lunch, test_preparation_course, COUNT(*) AS students, ROUND(AVG(math_score),1) AS avg_math
FROM students GROUP BY lunch, test_preparation_course ORDER BY avg_math DESC;
