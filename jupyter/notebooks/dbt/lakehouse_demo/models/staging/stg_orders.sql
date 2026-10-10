select
    order_id,
    customer_id,
    product_id,
    quantity,
    order_ts,
    case status
        when 'placed'   then 'open'
        when 'shipped'  then 'fulfilled'
        when 'returned' then 'returned'
        else 'unknown'
    end as status
from {{ ref('raw_orders') }}
