-- get the current datetime
SELECT GETDATE();

-- Count the Number of records in a table
SELECT COUNT(*) FROM table1;

-- find the names of employees that begin with 'A'
SELECT * FROM table_name WHERE EmpName like 'A%'

-- get the 3rd highest salary from table
SELECT TOP 1 salary
FROM(
    SELECT TOP 3 salary
    FROM employee_table
    ORDER BY salary DESC) AS emp 
ORDER BY salary ASC;


-- Get the median from a table
SET @rowindex := -1;
 
SELECT
   AVG(d.distance) as Median 
FROM
   (SELECT @rowindex:=@rowindex + 1 AS rowindex,
           demo.distance AS distance
    FROM demo
    ORDER BY demo.distance) AS d
WHERE
d.rowindex IN (FLOOR(@rowindex / 2), CEIL(@rowindex / 2));


-- join on a table with a range of keys
SELECT 
    CASE WHEN g.Grade >= 8 THEN s.Name ELSE NULL END 'Name',
    g.Grade,
    s.Marks
FROM Students s
INNER JOIN Grades g
    ON s.Marks BETWEEN g.Min_Mark AND g.Max_Mark
ORDER BY 
    g.Grade DESC, CASE WHEN g.Grade >= 8 THEN s.Name ELSE CAST(s.Marks as VARCHAR(100)) END



-- Min and Max length of string 
SELECT
    STATION.CITY
    ,LENGTH(STATION.CITY) AS L
FROM STATION
WHERE LENGTH(STATION.CITY) = 
        LENGTH(STATION.CITY) = (SELECT MIN(LENGTH(STATION.CITY)) FROM STATION)
--ORDER BY L ASC, STATION.CITY ASC
UNION
SELECT
    STATION.CITY
    ,LENGTH(STATION.CITY) AS L
FROM STATION
WHERE LENGTH(STATION.CITY) = 
        (SELECT MAX(LENGTH(STATION.CITY)) FROM STATION)
--ORDER BY L ASC, STATION.CITY ASC

(select CITY, length(CITY) from STATION order by length(CITY), CITY limit 1)
UNION
(select CITY, length(CITY) from STATION order by length(CITY) DESC limit 1)


-- check if a city has a name that starts with and ends with a vowel - NO DUPLICATES

    
SELECT DISTINCT STATION.CITY
FROM STATION
WHERE
    LOWER(STATION.CITY) LIKE '%a'
    or LOWER(STATION.CITY) LIKE '%e'
    or LOWER(STATION.CITY) LIKE '%i'
    or LOWER(STATION.CITY) LIKE '%o'
    or LOWER(STATION.CITY) LIKE '%u'
order by
    STATION.CITY
