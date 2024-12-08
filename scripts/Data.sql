DROP TABLE IF EXISTS melted_gdp_per_capita;
SELECT * FROM melted_gdp_per_capita;

SELECT *
FROM melted_gdp_per_capita
WHERE "Country Name" IN ('United States', 'China', 'India', 'United Kingdom');

DROP TABLE IF EXISTS melted_gdp_growth;
SELECT * FROM melted_gdp_growth;

SELECT *
FROM melted_gdp_growth
WHERE "Country Name" IN ('United States');

DROP TABLE IF EXISTS employment_to_population_ratio;
SELECT * FROM employment_to_population_ratio;

SELECT * FROM employment_to_population_ratio 
WHERE "Country_Name" = 'United States';


SELECT
    gdp_per_capita."Country Name",
    gdp_per_capita."Country Code",
    gdp_per_capita."Year",
    gdp_per_capita."GDP per Capita",
    gdp_growth."GDP Growth"
FROM
    melted_gdp_per_capita AS gdp_per_capita
LEFT JOIN
    melted_gdp_growth AS gdp_growth
ON
    gdp_per_capita."Country Code" = gdp_growth."Country Code"
    AND gdp_per_capita."Year" = gdp_growth."Year";

SELECT 
    gdp_growth."Country Name",
    gdp_growth."Country Code",
    gdp_growth."Year",
    gdp_growth."GDP Growth",
    employment."Employment_to_Population_Ratio"
FROM 
    melted_gdp_growth AS gdp_growth
LEFT JOIN 
    employment_to_population_ratio AS employment
ON 
    gdp_growth."Country Code" = employment."Country_Code"
    AND gdp_growth."Year" = employment."Year";


SELECT 
    gdp_per_capita."Country Name",
    gdp_per_capita."Country Code",
    gdp_per_capita."Year",
    gdp_per_capita."GDP per Capita",
    employment."Employment_to_Population_Ratio"
FROM 
    melted_gdp_per_capita AS gdp_per_capita
LEFT JOIN 
    employment_to_population_ratio AS employment
ON 
    gdp_per_capita."Country Code" = employment."Country_Code"
    AND gdp_per_capita."Year" = employment."Year";








