-- One row per payment. An order can be paid in several parts (e.g. a gift card plus a card).
select payment_id, order_id, method, amount, paid_at
from {{ ref('stg_payments') }}
