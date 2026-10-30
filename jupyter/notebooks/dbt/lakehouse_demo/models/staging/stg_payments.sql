select
    payment_id,
    order_id,
    method,
    cast(amount as double) as amount,
    paid_at
from {{ ref('raw_payments') }}
