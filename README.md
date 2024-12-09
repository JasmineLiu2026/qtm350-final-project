# QTM350 Final Project: Economic Indicators and Income Groups

This repository contains the final project for QTM350, analyzing the relationships between GDP growth, GDP per capita, and employment-to-population ratios across different income groups.

## **Group Members**  
- Jasmine Liu  
- Harris Wang  
- William Xu  
- Karl Zhou  

---

## **Research Questions**
1. How do GDP growth rates differ across all income groups?  
2. Does a higher employment-to-population ratio always correlate with higher GDP per capita in low-income countries?

---

## **Project Overview**  
Our analysis leverages real-world data from the **World Bank's World Development Indicators (WDI)** database. Specifically, we focus on the following key economic indicators:  
- **GDP Per Capita**  
- **GDP Growth (Annual %)**  
- **Employment-to-Population Ratio**  

The study examines differences across four income groups:  
- High-income  
- Upper-middle-income  
- Lower-middle-income  
- Low-income  

---

## **Repository Structure**
### **1. Data**  
- **Datasets**: Found in the `dataset` folder. These include the raw datasets used for analysis.

### **2. Scripts**  
- **`Data.sql`**:  
  Contains data queries and merging steps.
- **`data_preprocessing.ipynb`**:  
  Contains steps for data cleaning, merging, and exploratory analysis, including summary statistics.  
- **`data_analysis.py`**:  
  Builds on preprocessing with detailed analysis and visualizations addressing the research questions.

### **3. Outputs**  
- **Figures and Tables**:  
  Generated figures and tables are stored in the `figures` folder.

### **4. Documentation**  
- **Codebooks**:  
  Detailed descriptions of each dataset are provided in the `documentation` folder.  
- **Codebook Generator**:  
  Use `codebook.ipynb` to replicate codebook files.

---

## **Reproducing the Project**
To replicate the analysis, follow these steps:

1. **Download Datasets**:  
   Access the datasets from the `dataset` folder.  

2. **Run the Analysis**:  
   - Open the `data_analysis.py` file in your preferred Python IDE.  
   - Execute the script, which includes preprocessing and analysis steps.  

3. **Review Outputs**:  
   - Visualizations and summary tables will be generated and saved in the `figures` folder.

---

## **Technologies Used**  
- **SQL**: Data Queries
- **Python**: Data analysis and visualization  
- **Jupyter Notebook**: Documentation and codebook generation  
- **Pandas & NumPy**: Data preprocessing and analysis
- **Matplotlib & Seaborn**: Data visualization
- **Quarto**: Report Compilation

---

## **Acknowledgments**  
We extend our gratitude to the **World Bank** for providing the datasets and to our course instructor, Danilo Freire, for his guidance throughout this project.
