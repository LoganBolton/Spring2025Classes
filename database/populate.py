import csv
import os
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime # Import datetime

# Database connection configuration
config = {
    'user': 'ldb0046',
    'password': '...',
    'host': 'sysmysql8.auburn.edu',
    'database': 'ldb0046db',
    # Consider adding charset for robustness, though often defaults work
    # 'charset': 'utf8mb4'
}

# Establish connection to MySQL
def connect_to_database():
    try:
        conn = mysql.connector.connect(**config)
        print("Database connection successful.")
        return conn
    except mysql.connector.Error as err:
        if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Something is wrong with your user name or password")
        elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Database does not exist")
        else:
            print(f"Error connecting to the database: {err}")
        return None

# Function to drop tables if they exist
def drop_tables(conn, table_names):
    cursor = conn.cursor()
    # Disable foreign key checks before dropping
    cursor.execute("SET FOREIGN_KEY_CHECKS=0;")
    print("Dropping existing tables...")
    for table in reversed(table_names): # Drop in reverse order (child first)
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
            print(f"Table {table} dropped successfully.")
        except mysql.connector.Error as err:
            print(f"Error dropping table {table}: {err}")
    # Re-enable foreign key checks
    cursor.execute("SET FOREIGN_KEY_CHECKS=1;")
    conn.commit()
    cursor.close()

# Function to create tables
def create_tables(conn):
    cursor = conn.cursor()
    print("Creating tables...")
    try:
        # Create db_subject Table
        cursor.execute("""
        CREATE TABLE db_subject (
            SubjectID INT PRIMARY KEY AUTO_INCREMENT,
            CategoryName VARCHAR(100) NOT NULL
        );
        """)

        # Create db_supplier Table
        cursor.execute("""
        CREATE TABLE db_supplier (
            SupplierID INT PRIMARY KEY AUTO_INCREMENT,
            CompanyName VARCHAR(100) NOT NULL,
            ContactLastName VARCHAR(50),
            ContactFirstName VARCHAR(50),
            Phone VARCHAR(20)
        );
        """)

        # Create db_employee Table
        cursor.execute("""
        CREATE TABLE db_employee (
            EmployeeID INT PRIMARY KEY AUTO_INCREMENT,
            LastName VARCHAR(50) NOT NULL,
            FirstName VARCHAR(50) NOT NULL
        );
        """)

        # Create db_customer Table
        cursor.execute("""
        CREATE TABLE db_customer (
            CustomerID INT PRIMARY KEY AUTO_INCREMENT,
            LastName VARCHAR(50) NOT NULL,
            FirstName VARCHAR(50) NOT NULL,
            Phone VARCHAR(20)
        );
        """)

        # Create db_shipper Table
        cursor.execute("""
        CREATE TABLE db_shipper (
            ShipperID INT PRIMARY KEY AUTO_INCREMENT,
            ShipperName VARCHAR(100) NOT NULL
        );
        """)

        # Create db_book Table
        cursor.execute("""
        CREATE TABLE db_book (
            BookID INT PRIMARY KEY AUTO_INCREMENT,
            Title VARCHAR(150) NOT NULL,
            UnitPrice DECIMAL(10, 2) NOT NULL,
            Author VARCHAR(100),
            Quantity INT NOT NULL,
            SupplierID INT,
            SubjectID INT,
            FOREIGN KEY (SupplierID) REFERENCES db_supplier(SupplierID) ON DELETE SET NULL ON UPDATE CASCADE,
            FOREIGN KEY (SubjectID) REFERENCES db_subject(SubjectID) ON DELETE SET NULL ON UPDATE CASCADE
        );
        """) # Added ON DELETE/UPDATE actions for FKs - adjust as needed

        # Create db_order Table
        cursor.execute("""
        CREATE TABLE db_order (
            OrderID INT PRIMARY KEY,  # Keep explicit OrderID from CSV
            CustomerID INT NOT NULL,
            EmployeeID INT NOT NULL,
            OrderDate DATE NOT NULL,  # This column caused errors
            ShippedDate DATE DEFAULT NULL, # This column might cause errors
            ShipperID INT,
            FOREIGN KEY (CustomerID) REFERENCES db_customer(CustomerID) ON DELETE RESTRICT ON UPDATE CASCADE,
            FOREIGN KEY (EmployeeID) REFERENCES db_employee(EmployeeID) ON DELETE RESTRICT ON UPDATE CASCADE,
            FOREIGN KEY (ShipperID) REFERENCES db_shipper(ShipperID) ON DELETE SET NULL ON UPDATE CASCADE
        );
        """) # Added ON DELETE/UPDATE actions for FKs - adjust as needed

        # Create db_order_detail Table
        cursor.execute("""
        CREATE TABLE db_order_detail (
            OrderDetailID INT PRIMARY KEY AUTO_INCREMENT,
            OrderID INT NOT NULL,
            BookID INT NOT NULL,
            Quantity INT NOT NULL,
            FOREIGN KEY (OrderID) REFERENCES db_order(OrderID) ON DELETE CASCADE ON UPDATE CASCADE, # Cascade delete often makes sense here
            FOREIGN KEY (BookID) REFERENCES db_book(BookID) ON DELETE RESTRICT ON UPDATE CASCADE
        );
        """) # Added ON DELETE/UPDATE actions for FKs - adjust as needed

        print("All tables created successfully.")
    except mysql.connector.Error as err:
        print(f"Error creating tables: {err}")
    finally:
        if cursor:
            cursor.close()

# --- MODIFIED FUNCTION ---
# Function to read CSV files and insert data into the database
def insert_data_from_csv(conn, table_name, csv_file_path, columns, auto_increment_columns=[]):
    cursor = conn.cursor()
    columns_to_insert = [col for col in columns if col not in auto_increment_columns]
    placeholders = ", ".join(["%s"] * len(columns_to_insert))
    columns_str = ", ".join(f"`{col}`" for col in columns_to_insert) # Use backticks for safety
    sql = f"INSERT INTO `{table_name}` ({columns_str}) VALUES ({placeholders})"

    date_columns = {'OrderDate', 'ShippedDate'} # Define columns needing date conversion
    processed_count = 0
    error_count = 0

    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile: # Specify encoding
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                 print(f"Warning: CSV file {csv_file_path} might be empty or header is missing.")
                 return
            # Validate CSV headers match expected columns
            expected_cols_set = set(columns)
            csv_headers_set = set(reader.fieldnames)
            missing_cols = expected_cols_set - csv_headers_set
            extra_cols = csv_headers_set - expected_cols_set
            if missing_cols:
                 print(f"Warning: Missing columns in {csv_file_path}: {missing_cols}")
            # if extra_cols:
            #      print(f"Warning: Extra columns in {csv_file_path} not used by script: {extra_cols}")


            for row_num, row in enumerate(reader, start=1):
                values_list = []
                try:
                    for col in columns_to_insert:
                        raw_value = row.get(col, '').strip() # Use .get for safety, default to empty string

                        if raw_value.upper() == 'NULL' or raw_value == '':
                            values_list.append(None)
                        elif col in date_columns:
                             # Attempt to parse date in M/D/YYYY format
                             try:
                                 formatted_date = datetime.strptime(raw_value, '%m/%d/%Y').strftime('%Y-%m-%d')
                                 values_list.append(formatted_date)
                             except ValueError:
                                 print(f"Error parsing date value '{raw_value}' in column '{col}', file {csv_file_path}, row {row_num}. Inserting NULL.")
                                 values_list.append(None) # Insert NULL if date format is wrong
                        else:
                            values_list.append(raw_value)

                    values = tuple(values_list)
                    cursor.execute(sql, values)
                    processed_count += 1
                except mysql.connector.Error as err:
                    print(f"Error inserting data into {table_name} (Row {row_num}): {err}. Data: {values}")
                    error_count += 1
                except KeyError as e:
                    print(f"Error: Missing expected column '{e}' in {csv_file_path} at row {row_num}. Skipping row.")
                    error_count += 1

        conn.commit()
        print(f"Data insertion attempt finished for {table_name} from {csv_file_path}. Processed: {processed_count}, Errors: {error_count}.")

    except FileNotFoundError:
         print(f"Error: CSV file not found at {csv_file_path}")
    except Exception as e:
         print(f"An unexpected error occurred while processing {csv_file_path}: {e}")
    finally:
        if cursor:
            cursor.close()


# Main script to drop, recreate tables, and insert data
def main():
        conn = connect_to_database()
        if not conn:
            return

        # List of tables in creation order (dependencies first)
        # Dropping will happen in reverse order automatically in drop_tables
        tables_in_creation_order = [
            'db_subject',
            'db_supplier',
            'db_employee',
            'db_customer',
            'db_shipper',
            'db_book',
            'db_order',
            'db_order_detail'
        ]

        drop_tables(conn, tables_in_creation_order) # Pass the list to drop function

        create_tables(conn) # Create tables

        # Define your CSV files and their corresponding table information
        # Ensure the 'columns' list matches the exact headers in your CSV files
        # Specify which columns are auto-incremented by the DB
        csv_files_info = {
            'data/db_subject.csv': ('db_subject', ['SubjectID', 'CategoryName'], ['SubjectID']),
            'data/db_supplier.csv': ('db_supplier', ['SupplierID', 'CompanyName', 'ContactLastName', 'ContactFirstName', 'Phone'], ['SupplierID']),
            'data/db_employee.csv': ('db_employee', ['EmployeeID', 'LastName', 'FirstName'], ['EmployeeID']),
            'data/db_customer.csv': ('db_customer', ['CustomerID', 'LastName', 'FirstName', 'Phone'], ['CustomerID']),
            'data/db_shipper.csv': ('db_shipper', ['ShipperID', 'ShipperName'], ['ShipperID']), # CHECK THIS CSV FOR EMPTY ShipperName
            'data/db_book.csv': ('db_book', ['BookID', 'Title', 'UnitPrice', 'Author', 'Quantity', 'SupplierID', 'SubjectID'], ['BookID']),
            'data/db_order.csv': ('db_order', ['OrderID', 'CustomerID', 'EmployeeID', 'OrderDate', 'ShippedDate', 'ShipperID'], []),  # OrderID comes from CSV
            # Check db_order_detail.csv headers. Does it have OrderDetailID or just OrderID, BookID, Quantity?
            # Assuming it does NOT have OrderDetailID as it's AUTO_INCREMENT in the table
            'data/db_order_detail.csv': ('db_order_detail', ['OrderID', 'BookID', 'Quantity'], []) # OrderDetailID is auto-generated
        }

        # Insert data from CSV files in the correct order (respecting FK constraints)
        for table_name in tables_in_creation_order:
            # Find the corresponding CSV file info entry
            found = False
            for file_path, (tbl, cols, auto_inc_cols) in csv_files_info.items():
                if tbl == table_name:
                    print(f"Inserting data from {file_path} into {table_name}...")
                    insert_data_from_csv(conn, table_name, file_path, cols, auto_increment_columns=auto_inc_cols)
                    found = True
                    break
            if not found:
                print(f"Warning: No CSV info found for table {table_name} in csv_files_info dictionary.")


        print("--- Script Execution Finished ---")
        conn.close()

if __name__ == "__main__":
    main()