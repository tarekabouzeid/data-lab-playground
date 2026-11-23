select
    refund_id,
    order_id,
    cast(amount as double) as amount,
    refunded_at,
    reason
from {{ ref('raw_refunds') }}
