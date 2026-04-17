# Applied Data Science Project 3: Experimentation 

## Overview

This project evaluates how **instructional text affects user performance and experience** in a data analysis web application.

**Research Question:**
*Does clearer, more guided instructional text improve task completion, efficiency, and user experience?*

We compare:

* **Version A (Control):** minimal instructions
* **Version B (Treatment):** more guided instructional text

---

## Repository Structure

```
.
├── Analysis/
│   ├── project3_analysis.py
│   └── (figures and results)
├── App_Versions/
│   ├── app_simple_ab.py   # Control (A)
│   └── app_original.py    # Treatment (B)
├── Data/
│   ├── App_Ver_A_Data_raw.csv # Control Dataset (A)
│   └── App_Ver_B_Data_raw.csv # Treatment Dataset (B)
├── logs/
├── Project 3.pdf
├── README.md
└── requirements.txt
```

* **Analysis/** → A/B test code and visual results
* **App_Versions/** → both app versions
* **Data/** → Google Form response data

---

## Data Collected

After using the app, users reported:

* **Time (seconds)** *(1 min = 60 sec)*
* **Ease** *(1 = Difficult, 7 = Easy)*
* **Clarity** *(1 = Unclear, 7 = Clear)*
* **Guidance** *(1 = None, 7 = High)*
* **Completion Level**

  * 0 = No Progress
  * 1 = Explored Data
  * 2 = Generated Chart

These capture:

* task completion
* engagement
* efficiency
* usability

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Control (Version A)

```bash
python -m shiny run App_Versions/app_simple_ab.py
```

### Run Treatment (Version B)

```bash
python -m shiny run App_Versions/app_original.py
```

### Run Analysis

```bash
python Analysis/project3_analysis.py
```

---

## Summary

This repository contains:

* two experimental app versions
* user response data collected via Google Forms
* analysis comparing control vs treatment

**Goal:** determine whether guided instructions improve user outcomes in a data analysis interface.

## Team Members 

* Wentao Zhong (wz2753)
* Nikhil Shanbhag (nvs2128)
* Zhanhang Shi (zs2741)
* Xiying Chen (xc2781)
