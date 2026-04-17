# Applied Data Science Project 3: Experimentation 

\section*{Project Overview}

This project evaluates how \textbf{instructional design affects user performance and experience} in a data analysis web application through an A/B test.

\textbf{Research Question:}  
\emph{Does clearer, more guided instructional text improve task completion, efficiency, and user experience in a data analysis app?}

We compare two versions of the same Shiny app:
\begin{itemize}
    \item \textbf{Version A (Control):} minimal instructions
    \item \textbf{Version B (Treatment):} more guided instructional text and terminology
\end{itemize}

\section*{Repository Structure}

\begin{verbatim}
.
├── Analysis/
│   ├── project3_analysis.py
│   └── (figures and result visuals)
├── App_Versions/
│   ├── app_simple_matched.py   # Version A: Control
│   └── core.py                 # Version B: Treatment
├── Data/
│   ├── (control CSV files)
│   └── (treatment CSV files)
├── logs/
├── README.md
├── requirements.txt
└── Project 3.pdf
\end{verbatim}

\begin{itemize}
    \item \texttt{Analysis/}: main A/B test analysis code and figures summarizing results
    \item \texttt{App\_Versions/}: both app versions used in the experiment
    \item \texttt{Data/}: CSV files collected from Google Forms for both versions
\end{itemize}

\section*{Collected Data}

User responses were collected through Google Forms after interacting with one version of the app. The main variables are:

\begin{itemize}
    \item \textbf{Approximate Time Spent in Seconds}  
    (\emph{Note: 1 minute = 60 seconds})
    \item \textbf{Ease of Use}  
    \((1=\text{Difficult},\ 7=\text{Easy})\)
    \item \textbf{Clarity}  
    \((1=\text{Unclear},\ 7=\text{Clear})\)
    \item \textbf{Guidance Felt}  
    \((1=\text{None},\ 7=\text{High})\)
    \item \textbf{Level of Completion}  
    \((0=\text{No Progress},\ 1=\text{Explored Data},\ 2=\text{Generated a Chart or Visual})\)
\end{itemize}

These variables are used to measure:
\begin{itemize}
    \item task completion,
    \item engagement,
    \item efficiency,
    \item and perceived usability/clarity.
\end{itemize}

\section*{How to Run}

\textbf{Install dependencies:}
\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

\textbf{Run Version A (Control):}
\begin{verbatim}
python -m shiny run App_Versions/app_simple_matched.py
\end{verbatim}

\textbf{Run Version B (Treatment):}
\begin{verbatim}
python -m shiny run App_Versions/core.py
\end{verbatim}

\textbf{Run the analysis:}
\begin{verbatim}
python Analysis/project3_analysis.py
\end{verbatim}

\section*{Summary}

This repository contains the two app versions, the Google Form response data, and the analysis code used to compare the control and treatment groups. The goal is to determine whether more guided instructional text leads to better user outcomes in a data analysis interface.
