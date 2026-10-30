{#
  Incremental model on an Iceberg table: `merge` strategy = one MERGE INTO per run (one Iceberg snapshot per run).
  Re-reads a 1-day look-back so late status changes of recent orders update the existing rows.
#}
{{
  config(
    materialized='incremental',
    unique_key='order_id',
    incremental_strategy='merge',
    on_schema_change='sync_all_columns',
    properties={"partitioning": "ARRAY['day(order_ts)']"}
  )
}}

select
    o.order_id,
    o.customer_id,
    o.product_id,
    o.quantity,
    p.unit_price,
    o.quantity * p.unit_price as amount,
    o.order_ts,
    o.status
from {{ ref('stg_orders') }} as o
join {{ ref('stg_products') }} as p on p.product_id = o.product_id
{% if is_incremental() %}
where o.order_ts >= (select max(order_ts) - interval '1' day from {{ this }})
{% endif %}
