SELECT DISTINCT
    primary_type,
    location_description,
    domestic,
    arrest
FROM `bigquery-public-data.chicago_crime.crime`
WHERE year > @year

