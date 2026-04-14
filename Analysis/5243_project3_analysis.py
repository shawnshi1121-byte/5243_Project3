import re
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

# =========================
# File paths
# =========================
FILE_A = 'App_Ver_A_Data.csv'
FILE_B = 'App_Ver_B_Data.csv'

# =========================
# Original column names
# =========================
TIME_COL = 'Approximate Time Spent in Seconds (Note: 1 minute = 60 seconds)'
EASE_COL = 'Ease of use (1=Difficult, 7=Easy)'
CLARITY_COL = 'Clarity (1=Unclear, 7=Clear)'
GUIDANCE_COL = 'Guidance felt (1=None, 7=High)'
COMPLETION_COL = 'Level of Completion (0 = No Progress, 1 = Explored Data, 2 = Generated a Chart or Visual)'

# =========================
# Cleaning helper
# =========================
def clean_time_entry(value, small_threshold_minutes=15):
    """
    Convert the raw time entry into seconds.

    Rules:
    1. If the entry explicitly contains 'minute' or 'min', convert to seconds.
    2. If it explicitly contains 'second' or 'sec', keep as seconds.
    3. If it is a very small pure number (<= 15), assume the participant likely entered minutes,
       because the survey asked for seconds and such tiny values are suspicious.
    4. Otherwise keep the numeric value as seconds.

    Returns:
        cleaned_seconds, cleaning_note, suspicious_flag
    """
    raw = str(value).strip().lower()
    match = re.search(r'(\d+(?:\.\d+)?)', raw)

    if match is None:
        return np.nan, 'unreadable_time_entry', True

    number = float(match.group(1))

    if re.search(r'min', raw):
        return number * 60, 'explicit_minutes_to_seconds', True

    if re.search(r'sec', raw):
        return number, 'explicit_seconds_kept', False

    if number <= small_threshold_minutes:
        return number * 60, 'small_numeric_assumed_minutes', True

    return number, 'numeric_as_seconds', False


def load_and_clean(file_path, version_label):
    df = pd.read_csv(file_path)
    df['version'] = version_label
    df['time_raw'] = df[TIME_COL].astype(str).str.strip()

    cleaned = df['time_raw'].apply(clean_time_entry)
    df[['time_spent_seconds', 'time_cleaning_note', 'suspicious_time_flag']] = pd.DataFrame(
        cleaned.tolist(), index=df.index
    )

    # Standardize column names for easier analysis
    df = df.rename(columns={
        EASE_COL: 'ease_of_use',
        CLARITY_COL: 'clarity',
        GUIDANCE_COL: 'guidance_felt',
        COMPLETION_COL: 'completion_level'
    })

    # Convert analysis columns to numeric if needed
    numeric_cols = ['ease_of_use', 'clarity', 'guidance_felt', 'completion_level', 'time_spent_seconds']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def cohens_d(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    if pooled == 0:
        return np.nan
    return (np.mean(y) - np.mean(x)) / pooled


def cramers_v_from_chi2(chi2, n, r, c):
    return np.sqrt(chi2 / (n * min(r - 1, c - 1)))


def summarize_by_version(df):
    summary = df.groupby('version')[['time_spent_seconds', 'ease_of_use', 'clarity', 'guidance_felt', 'completion_level']].agg(
        ['count', 'mean', 'std', 'median', 'min', 'max']
    )
    return summary.round(3)


def run_continuous_test(df, variable_name):
    a = df.loc[df['version'] == 'A', variable_name].dropna()
    b = df.loc[df['version'] == 'B', variable_name].dropna()

    # Welch's t-test (safe when variances may differ)
    t_res = stats.ttest_ind(a, b, equal_var=False)

    # Mann-Whitney U as a nonparametric robustness check
    mw_res = stats.mannwhitneyu(a, b, alternative='two-sided')

    return {
        'variable': variable_name,
        'mean_A': a.mean(),
        'mean_B': b.mean(),
        'median_A': a.median(),
        'median_B': b.median(),
        'welch_t_stat': t_res.statistic,
        'welch_p_value': t_res.pvalue,
        'mannwhitney_u': mw_res.statistic,
        'mannwhitney_p_value': mw_res.pvalue,
        'cohens_d_B_minus_A': cohens_d(a, b)
    }


def run_completion_test(df):
    table = pd.crosstab(df['version'], df['completion_level'])
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v_from_chi2(chi2, table.to_numpy().sum(), *table.shape)
    return table, {
        'chi2_stat': chi2,
        'p_value': p,
        'degrees_of_freedom': dof,
        'cramers_v': v
    }


def save_plots(df):
    plot_vars = {
        'time_spent_seconds': 'Time Spent (seconds)',
        'ease_of_use': 'Ease of Use',
        'clarity': 'Clarity',
        'guidance_felt': 'Guidance Felt',
        'completion_level': 'Completion Level'
    }

    for var, title in plot_vars.items():
        plt.figure(figsize=(7, 5))
        df.boxplot(column=var, by='version')
        plt.title(f'{title} by Version')
        plt.suptitle('')
        plt.xlabel('Version')
        plt.ylabel(title)
        plt.tight_layout()
        plt.savefig(f'{var}_boxplot.png', dpi=300)
        plt.close()


def main():
    # Load and clean
    df_a = load_and_clean(FILE_A, 'A')
    df_b = load_and_clean(FILE_B, 'B')
    df = pd.concat([df_a, df_b], ignore_index=True)

    # Save cleaned dataset for transparency / reproducibility
    df.to_csv('cleaned_ab_test_data.csv', index=False)

    # Export suspicious records for manual review
    suspicious = df.loc[df['suspicious_time_flag'], ['version', 'Timestamp', 'time_raw', 'time_spent_seconds', 'time_cleaning_note']]
    suspicious.to_csv('suspicious_time_entries_to_review.csv', index=False)

    print('\n=========================')
    print('CLEANING SUMMARY')
    print('=========================')
    print(df['time_cleaning_note'].value_counts(dropna=False))

    print('\nSuspicious / manually reviewable time entries:')
    print(suspicious.to_string(index=False))

    print('\n=========================')
    print('DESCRIPTIVE STATISTICS')
    print('=========================')
    print(summarize_by_version(df))

    print('\n=========================')
    print('INDEPENDENT GROUP TESTS')
    print('=========================')
    continuous_vars = ['time_spent_seconds', 'ease_of_use', 'clarity', 'guidance_felt']
    results = []
    for var in continuous_vars:
        res = run_continuous_test(df, var)
        results.append(res)
        print(f"\nVariable: {var}")
        for k, v in res.items():
            if k != 'variable':
                print(f'  {k}: {v}')

    results_df = pd.DataFrame(results)
    results_df.to_csv('continuous_test_results.csv', index=False)

    print('\n=========================')
    print('COMPLETION LEVEL ANALYSIS')
    print('=========================')
    completion_table, completion_test = run_completion_test(df)
    print('\nCompletion contingency table:')
    print(completion_table)
    print('\nChi-square results:')
    for k, v in completion_test.items():
        print(f'  {k}: {v}')

    pd.DataFrame([completion_test]).to_csv('completion_test_results.csv', index=False)
    completion_table.to_csv('completion_contingency_table.csv')

    # Optional composite satisfaction/usability score
    df['usability_index'] = df[['ease_of_use', 'clarity', 'guidance_felt']].mean(axis=1)
    usability_res = run_continuous_test(df, 'usability_index')
    print('\n=========================')
    print('OPTIONAL COMPOSITE INDEX')
    print('=========================')
    print('Usability index = mean(ease_of_use, clarity, guidance_felt)')
    for k, v in usability_res.items():
        if k != 'variable':
            print(f'  {k}: {v}')
    pd.DataFrame([usability_res]).to_csv('usability_index_test_results.csv', index=False)

    # Save plots
    save_plots(df)

    print('\nFiles created:')
    print('- cleaned_ab_test_data.csv')
    print('- suspicious_time_entries_to_review.csv')
    print('- continuous_test_results.csv')
    print('- completion_contingency_table.csv')
    print('- completion_test_results.csv')
    print('- usability_index_test_results.csv')
    print('- *_boxplot.png files')


if __name__ == '__main__':
    main()

