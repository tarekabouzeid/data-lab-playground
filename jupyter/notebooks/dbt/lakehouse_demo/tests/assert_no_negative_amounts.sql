-- singular test: returns the rows that violate the rule; the test passes when no rows come back
select order_id, amount from {{ ref('fct_orders') }} where amount < 0
