custom_css = """
/* Limit the width of the first AutoEvalColumn so that names don't expand too much */
table td:first-child,
table th:first-child {
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
}

/* Full width space */
.gradio-container {
    max-width: 95% !important;
}

/* Text style and margins */
.markdown-text {
    font-size: 16px !important;
}

#models-to-add-text {
    font-size: 18px !important;
}

#citation-button span {
    font-size: 16px !important;
}

#citation-button textarea {
    font-size: 16px !important;
}

#citation-button > label > button {
    margin: 6px;
    transform: scale(1.3);
}

#search-bar-table-box > div:first-child {
    background: none;
    border: none;
}

#search-bar {
    padding: 0px;
}

.tab-buttons button {
    font-size: 20px;
}

/* Filters style */
#filter_type {
    border: 0;
    padding-left: 0;
    padding-top: 0;
}
#filter_type label {
    display: flex;
}
#filter_type label > span {
    margin-top: var(--spacing-lg);
    margin-right: 0.5em;
}
#filter_type label > .wrap {
    width: 103px;
}
#filter_type label > .wrap .wrap-inner {
    padding: 2px;
}
#filter_type label > .wrap .wrap-inner input {
    width: 1px;
}
#filter-columns-type {
    border: 0;
    padding: 0.5;
}
#filter-columns-size {
    border: 0;
    padding: 0.5;
}
#box-filter > .form {
    border: 0;
}

/* Header styles */
#header-title {
    text-align: left;
    display: inline-block;
}

#header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

#header-row .gradio-html {
    flex-grow: 1;
}

#oauth-button {
    height: auto;
    min-width: max-content;
    white-space: nowrap;
    padding: 10px 20px;
    border-radius: 4px;
}
"""

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
    }
    """