{# Business rule shared by models and covered by a unit test. #}
{% macro spend_tier(amount_expr) -%}
    case
        when {{ amount_expr }} >= 2000 then 'gold'
        when {{ amount_expr }} >= 500  then 'silver'
        else 'bronze'
    end
{%- endmacro %}
