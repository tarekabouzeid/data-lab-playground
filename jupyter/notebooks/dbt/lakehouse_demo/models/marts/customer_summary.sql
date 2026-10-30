select
    c.customer_id,
    c.name,
    c.country,
    c.plan,
    count(f.order_id)                       as orders,
    coalesce(sum(f.amount), 0)              as total_spend,
    {{ spend_tier('coalesce(sum(f.amount), 0)') }} as spend_tier
from {{ ref('stg_customers') }} as c
left join {{ ref('fct_orders') }} as f on f.customer_id = c.customer_id and f.status <> 'returned'
group by 1, 2, 3, 4
