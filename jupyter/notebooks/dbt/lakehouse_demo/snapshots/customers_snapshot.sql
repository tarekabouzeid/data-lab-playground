{% snapshot customers_snapshot %}
{{
  config(
    target_schema='dbt_demo_snapshots',
    unique_key='customer_id',
    strategy='check',
    check_cols=['plan', 'country']
  )
}}
select customer_id, name, country, plan from {{ ref('raw_customers') }}
{% endsnapshot %}
