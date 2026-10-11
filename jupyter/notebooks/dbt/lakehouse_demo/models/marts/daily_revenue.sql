{{ config(properties={"partitioning": "ARRAY['month(order_date)']"}) }}

select
    cast(order_ts as date) as order_date,
    count(*)               as orders,
    sum(amount)            as revenue
from {{ ref('fct_orders') }}
where status <> 'returned'
group by 1
