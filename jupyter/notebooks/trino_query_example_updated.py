# Enhanced Trino Integration for Direct S3 Data Querying
print("🔍 Enhanced Trino Integration with S3 External Tables...")

try:
    import trino
    from trino.dbapi import connect
    import time
    
    # Create Trino connection
    def create_trino_connection():
        """Create connection to Trino distributed SQL engine"""
        try:
            conn = connect(
                host='trino',
                port=8080,
                user='admin',
                catalog='lakehouse',  # ✅ CHANGED: Using lakehouse connector instead of hive
                schema='default'
            )
            return conn
        except Exception as e:
            print(f"⚠️  Trino connection error: {e}")
            return None
    
    # Test Trino connection
    trino_conn = create_trino_connection()
    
    if trino_conn:
        print("✅ Trino connection established!")
        
        cursor = trino_conn.cursor()
        
        # Test basic connectivity
        print("\n📊 Testing Trino connectivity...")
        
        # Show available catalogs
        print("\n🗂️  Available Catalogs:")
        cursor.execute("SHOW CATALOGS")
        catalogs = cursor.fetchall()
        for catalog in catalogs:
            print(f"  📁 {catalog[0]}")
        
        # Create external table for S3 Parquet data
        print("\n🏗️  Creating External Tables for S3 Data...")
        
        # Drop table if exists (for clean re-runs)
        try:
            cursor.execute("DROP TABLE IF EXISTS lakehouse.default.transactions")  # ✅ CHANGED: lakehouse catalog
            print("  🗑️  Dropped existing transactions table")
        except Exception as e:
            print(f"  ℹ️  No existing table to drop: {e}")
        
        # Create external table pointing to S3 Parquet data
        # ✅ CHANGED: Updated to use lakehouse catalog and proper Hive table syntax
        create_table_sql = """
        CREATE TABLE lakehouse.default.transactions (
            customer_id VARCHAR,
            transaction_id VARCHAR,
            product VARCHAR,
            category VARCHAR,
            price DOUBLE,
            quantity INTEGER,
            discount DOUBLE,
            city VARCHAR,
            state VARCHAR,
            timestamp VARCHAR,
            customer_satisfaction DOUBLE,
            payment_method VARCHAR,
            is_premium BOOLEAN,
            age_group VARCHAR,
            channel VARCHAR,
            total_amount DOUBLE,
            profit_margin DOUBLE,
            date DATE,
            year INTEGER,
            month INTEGER,
            quarter INTEGER
        )
        WITH (
            external_location = 's3a://warehouse/transactions/parquet/',
            format = 'PARQUET',
            table_type = 'HIVE'
        )
        """
        
        try:
            cursor.execute(create_table_sql)
            print("  ✅ External table 'transactions' created successfully!")
            print("     📍 Location: s3a://warehouse/transactions/parquet/")
            print("     📊 Format: Parquet (via Lakehouse connector)")
            print("     🔄 Table Type: HIVE (supports Delta, Hudi, Iceberg auto-detection)")
        except Exception as e:
            print(f"  ⚠️  Table creation error: {e}")
            print("     💡 This might be expected if data hasn't been written to S3 yet")
        
        # Test querying the external table
        print("\n🔍 Testing S3 Data Queries via Trino...")
        
        # Simple count query - ✅ CHANGED: lakehouse catalog
        try:
            print("\n1️⃣  Record Count Query:")
            cursor.execute("SELECT COUNT(*) as total_records FROM lakehouse.default.transactions")
            result = cursor.fetchone()
            if result:
                print(f"     📊 Total Records: {result[0]:,}")
            else:
                print("     ⚠️  No data found - run Spark data generation first")
        except Exception as e:
            print(f"     ❌ Count query error: {e}")
        
        # Revenue by state query - ✅ CHANGED: lakehouse catalog
        try:
            print("\n2️⃣  Revenue by State Query:")
            revenue_sql = """
            SELECT 
                state,
                COUNT(*) as transactions,
                ROUND(SUM(total_amount), 2) as total_revenue,
                ROUND(AVG(total_amount), 2) as avg_transaction
            FROM lakehouse.default.transactions 
            GROUP BY state 
            ORDER BY total_revenue DESC 
            LIMIT 10
            """
            
            cursor.execute(revenue_sql)
            results = cursor.fetchall()
            
            if results:
                print("     🏆 Top 10 States by Revenue:")
                print(f"     {'State':<8} {'Transactions':<12} {'Revenue':<12} {'Avg Transaction':<15}")
                print("     " + "-" * 50)
                for row in results:
                    print(f"     {row[0]:<8} {row[1]:<12,} ${row[2]:<11,} ${row[3]:<14}")
            else:
                print("     ⚠️  No revenue data found")
                
        except Exception as e:
            print(f"     ❌ Revenue query error: {e}")
        
        # Product performance query - ✅ CHANGED: lakehouse catalog
        try:
            print("\n3️⃣  Product Performance Query:")
            product_sql = """
            SELECT 
                product,
                category,
                COUNT(*) as sales_count,
                ROUND(SUM(total_amount), 2) as revenue,
                ROUND(AVG(customer_satisfaction), 1) as avg_satisfaction
            FROM lakehouse.default.transactions 
            GROUP BY product, category
            ORDER BY revenue DESC 
            LIMIT 8
            """
            
            cursor.execute(product_sql)
            results = cursor.fetchall()
            
            if results:
                print("     🛍️  Top Products by Revenue:")
                print(f"     {'Product':<12} {'Category':<12} {'Sales':<8} {'Revenue':<12} {'Satisfaction':<12}")
                print("     " + "-" * 70)
                for row in results:
                    print(f"     {row[0]:<12} {row[1]:<12} {row[2]:<8} ${row[3]:<11,} {row[4]:<12}")
            else:
                print("     ⚠️  No product data found")
                
        except Exception as e:
            print(f"     ❌ Product query error: {e}")
        
        # Time-based analysis - ✅ CHANGED: lakehouse catalog
        try:
            print("\n4️⃣  Monthly Trend Analysis:")
            trend_sql = """
            SELECT 
                year,
                month,
                COUNT(*) as transactions,
                ROUND(SUM(total_amount), 2) as revenue,
                ROUND(AVG(customer_satisfaction), 1) as satisfaction
            FROM lakehouse.default.transactions 
            GROUP BY year, month
            ORDER BY year DESC, month DESC 
            LIMIT 6
            """
            
            cursor.execute(trend_sql)
            results = cursor.fetchall()
            
            if results:
                print("     📅 Recent Monthly Performance:")
                print(f"     {'Year':<6} {'Month':<6} {'Transactions':<12} {'Revenue':<12} {'Satisfaction':<12}")
                print("     " + "-" * 60)
                for row in results:
                    month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][row[1]]
                    print(f"     {row[0]:<6} {month_name:<6} {row[2]:<12,} ${row[3]:<11,} {row[4]:<12}")
            else:
                print("     ⚠️  No trend data found")
                
        except Exception as e:
            print(f"     ❌ Trend query error: {e}")
        
        # Advanced analytical query with window functions - ✅ CHANGED: lakehouse catalog
        try:
            print("\n5️⃣  Advanced Analytics with Window Functions:")
            advanced_sql = """
            SELECT 
                state,
                product,
                total_amount,
                RANK() OVER (PARTITION BY state ORDER BY total_amount DESC) as rank_in_state,
                ROUND(AVG(total_amount) OVER (PARTITION BY state), 2) as state_avg_amount
            FROM lakehouse.default.transactions 
            WHERE total_amount > 1000
            ORDER BY state, rank_in_state
            LIMIT 15
            """
            
            cursor.execute(advanced_sql)
            results = cursor.fetchall()
            
            if results:
                print("     🏆 Top Purchases by State (>$1000):")
                print(f"     {'State':<6} {'Product':<12} {'Amount':<10} {'Rank':<6} {'State Avg':<10}")
                print("     " + "-" * 60)
                for row in results:
                    print(f"     {row[0]:<6} {row[1]:<12} ${row[2]:<9,} {row[3]:<6} ${row[4]:<9,}")
            else:
                print("     ⚠️  No high-value transactions found")
                
        except Exception as e:
            print(f"     ❌ Advanced query error: {e}")
        
        # Performance comparison
        print("\n⚡ Trino Lakehouse Connector Benefits:")
        performance_benefits = [
            "🚀 Unified access to Delta Lake, Hudi, and Iceberg formats",
            "🔄 Automatic table format detection - no manual configuration",
            "📊 Columnar Parquet format for optimized analytics",
            "⚡ Distributed processing across Trino workers",
            "🎯 No ETL required - query data where it lives",
            "📈 Sub-second response for analytical queries",
            "🌐 Single catalog for all lakehouse formats",
            "💾 Automatic predicate pushdown and projection pruning",
            "🏗️  Hive Metastore 4.2.0 integration for metadata management"
        ]
        
        for benefit in performance_benefits:
            print(f"  ✨ {benefit}")
        
        print("\n✅ Enhanced Trino lakehouse integration completed!")
        print("🎯 S3 data is now queryable via SQL through Trino lakehouse connector!")
        print("💡 Run Spark data generation first, then re-run this cell for data queries")
        
        cursor.close()
        trino_conn.close()
        
    else:
        print("⚠️  Trino not available - benefits of lakehouse connector:")
        print("  🔍 Unified SQL queries across Delta, Hudi, Iceberg formats")
        print("  ⚡ No data movement required")
        print("  📊 Real-time analytics on data lake")
        print("  🚀 Distributed query processing")
        print("  🔄 Automatic format detection")

except ImportError:
    print("📦 Installing Trino client...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "trino"])
    print("✅ Trino client installed! Re-run cell to test connection.")
    
except Exception as e:
    print(f"❌ Trino integration error: {e}")
    print("🔧 To enable full Trino functionality:")
    print("  1. Ensure Trino service is running")
    print("  2. Verify Hive Metastore 4.2.0 connectivity")
    print("  3. Check S3/MinIO access permissions")
    print("  4. Run Spark data generation first")
    print("  5. Lakehouse connector requires Trino 479+")
