-- Sales regions by customer country.
select country, sales_region, country_name
from {{ ref('raw_regions') }}
