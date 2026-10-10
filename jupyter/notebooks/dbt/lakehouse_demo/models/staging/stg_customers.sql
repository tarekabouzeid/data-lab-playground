select
    customer_id,
    name,
    lower(email) as email,
    country,
    plan,
    signup_date
from {{ ref('raw_customers') }}
