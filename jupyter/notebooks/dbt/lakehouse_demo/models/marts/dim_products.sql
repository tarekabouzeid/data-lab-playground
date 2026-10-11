{{ config(contract={"enforced": true}) }}

select
    cast(product_id as integer)  as product_id,
    cast(product_name as varchar) as product_name,
    cast(category as varchar)    as category,
    cast(unit_price as double)   as unit_price
from {{ ref('stg_products') }}
