select
    product_id,
    name as product_name,
    category,
    cast(unit_price as double) as unit_price
from {{ ref('raw_products') }}
