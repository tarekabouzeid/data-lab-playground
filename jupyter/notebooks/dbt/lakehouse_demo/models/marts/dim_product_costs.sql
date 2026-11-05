-- What the shop pays per unit, by product (the margin side of the catalogue).
select
    cast(product_id as integer) as product_id,
    cast(unit_cost as double)   as unit_cost,
    supplier
from {{ ref('raw_product_costs') }}
