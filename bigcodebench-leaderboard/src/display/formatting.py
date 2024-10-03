from huggingface_hub import HfApi

API = HfApi()


def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def make_clickable_model(df, model_col, link_col):
    df[model_col] = df.apply(
        lambda row: model_hyperlink(row[link_col], row[model_col]), axis=1
    )
    df["Openness"] = df.apply(
        lambda row: "Open" if "huggingface.co" in row[link_col] else "Closed", axis=1
    )
    return df


def styled_error(error):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{error}</p>"


def styled_warning(warn):
    return f"<p style='color: orange; font-size: 20px; text-align: center;'>{warn}</p>"


def styled_message(message):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{message}</p>"


def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)


def has_nan_values(df, columns):
    return df[columns].isna().any(axis=1)
