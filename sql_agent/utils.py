import re

def extract_sql_code(response: str) -> str:
    # Extract SQL block inside ```sql ... ```
    start = response.find('```sql\n')
    end = response.find('```', start + 7)

    if start != -1 and end != -1:
        sql_query = response[start + 7:end].strip()
    else:
        # If no ```sql ... ```, assume the response is plain text SQL
        sql_query = response.strip()

    # Remove "SQLQuery:" if present at the beginning
    sql_query = re.sub(r'^\s*SQLQuery:\s*', '', sql_query, flags=re.IGNORECASE)

    # Remove double quotes around identifiers
    sql_query = sql_query.replace('"', '')

    # Collapse multiple spaces/newlines into a single space
    sql_query = ' '.join(sql_query.split())

    return sql_query
