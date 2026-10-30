-- One row per refund. Returned orders are refunded (sometimes in two installments); some fulfilled orders get a
-- partial goodwill refund for damaged goods.
select refund_id, order_id, amount, refunded_at, reason
from {{ ref('stg_refunds') }}
