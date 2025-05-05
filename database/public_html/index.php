<?php
require_once('db_config.php');

$query = "";
$result = null;
$error = "";
$success = "";
$rowCount = 0;

if ($_SERVER["REQUEST_METHOD"] == "POST" && isset($_POST["query"])) {
    $query = trim($_POST["query"]);
    
    if (function_exists('get_magic_quotes_gpc') && get_magic_quotes_gpc()) {
        $query = stripslashes($query);
    }

    // Check if query contains DROP statement
    if (preg_match('/\bDROP\b/i', $query)) {
        $error = "DROP statements are not allowed.";
    } else if (!empty($query)) {
        $conn = connectDB();
        
        // Execute query
        try {
            $isSelect = (stripos($query, "SELECT") === 0);
            
            if ($isSelect) {
                $result = $conn->query($query);
                
                if ($result) {
                    $rowCount = $result->num_rows;
                    $success = "Query executed successfully. Rows retrieved: $rowCount";
                } else {
                    $error = "Error executing query: " . $conn->error;
                }
            } else {
                $result = $conn->query($query);
                
                if ($result) {
                    if (stripos($query, "INSERT") === 0) {
                        $success = "Row Inserted. Affected rows: " . $conn->affected_rows;
                    } else if (stripos($query, "UPDATE") === 0) {
                        $success = "Table Updated. Affected rows: " . $conn->affected_rows;
                    } else if (stripos($query, "DELETE") === 0) {
                        $success = "Row(s) Deleted. Affected rows: " . $conn->affected_rows;
                    } else if (stripos($query, "CREATE") === 0) {
                        $success = "Table Created.";
                    } else {
                        $success = "Query executed successfully.";
                    }
                } else {
                    $error = "Error executing query: " . $conn->error;
                }
            }
        } catch (Exception $e) {
            $error = "Error: " . $e->getMessage();
        }
        
        // Close connection
        $conn->close();
    }
}

// Define the numbered queries
$numberedQueries = array(
    // 1. Show the subject names of books supplied by supplier2
    "SELECT DISTINCT s.CategoryName 
    FROM db_subject s
    JOIN db_book b ON s.SubjectID = b.SubjectID
    JOIN db_supplier sup ON b.SupplierID = sup.SupplierID
    WHERE sup.CompanyName = 'supplier2'",

    // 2. Show the name and price of the most expensive book supplied by supplier3
    "SELECT b.Title, b.UnitPrice 
    FROM db_book b
    JOIN db_supplier s ON b.SupplierID = s.SupplierID
    WHERE s.CompanyName = 'supplier3'
    ORDER BY b.UnitPrice DESC
    LIMIT 1;",

    // 3. Show the unique names of all books ordered by lastname1 firstname1
    "SELECT DISTINCT b.Title
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    JOIN db_customer c ON o.CustomerID = c.CustomerID
    WHERE c.LastName = 'lastname1' AND c.FirstName = 'firstname1';",

    // 4. Show the title of books which have more than 10 units in stock
    "SELECT Title
    FROM db_book
    WHERE Quantity > 10;",

    // 5. Show the total price lastname1 firstname1 has paid for the books
    "SELECT SUM(od.Quantity * b.UnitPrice) AS TotalPaid
    FROM db_order_detail od
    JOIN db_book b ON od.BookID = b.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    JOIN db_customer c ON o.CustomerID = c.CustomerID
    WHERE c.LastName = 'lastname1' AND c.FirstName = 'firstname1';",

    // 6. Show the names of the customers who have paid less than $80 in totals
    "SELECT c.FirstName, c.LastName
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    JOIN db_book b ON od.BookID = b.BookID
    GROUP BY c.CustomerID, c.FirstName, c.LastName
    HAVING SUM(od.Quantity * b.UnitPrice) < 80;",

    // 7. Show the name of books supplied by supplier2
    "SELECT b.Title
    FROM db_book b
    JOIN db_supplier s ON b.SupplierID = s.SupplierID
    WHERE s.CompanyName = 'supplier2';",

    // 8. Show the total price each customer paid and their names. List the result in descending price
    "SELECT c.FirstName, c.LastName, SUM(od.Quantity * b.UnitPrice) AS TotalPaid
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    JOIN db_book b ON od.BookID = b.BookID
    GROUP BY c.CustomerID, c.FirstName, c.LastName
    ORDER BY TotalPaid DESC;",

    // 9. Show the names of all the books shipped on 08/04/2016 and their shippers' names
    "SELECT b.Title, s.ShipperName
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    JOIN db_shipper s ON o.ShipperID = s.ShipperID
    WHERE DATE(o.ShippedDate) = '2016-08-04';",

    // 10. Show the unique names of all the books lastname1 firstname1 and lastname4 firstname4 both ordered
    "SELECT DISTINCT b.Title
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    JOIN db_customer c ON o.CustomerID = c.CustomerID
    WHERE (c.LastName = 'lastname1' AND c.FirstName = 'firstname1')
    AND b.BookID IN (
        SELECT b2.BookID
        FROM db_book b2
        JOIN db_order_detail od2 ON b2.BookID = od2.BookID
        JOIN db_order o2 ON od2.OrderID = o2.OrderID
        JOIN db_customer c2 ON o2.CustomerID = c2.CustomerID
        WHERE c2.LastName = 'lastname4' AND c2.FirstName = 'firstname4'
    );",

    // 11. Show the names of all the books lastname6 firstname6 was responsible for
    "SELECT DISTINCT b.Title
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    JOIN db_employee e ON o.EmployeeID = e.EmployeeID
    WHERE e.LastName = 'lastname6' AND e.FirstName = 'firstname6';",

    // 12. Show the names of all the ordered books and their total quantities. List the result in ascending quantity
    "SELECT b.Title, SUM(od.Quantity) AS TotalQuantity
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    GROUP BY b.BookID, b.Title
    ORDER BY TotalQuantity ASC;",

    // 13. Show the names of the customers who ordered at least 2 books
    "SELECT c.FirstName, c.LastName
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    GROUP BY c.CustomerID, c.FirstName, c.LastName
    HAVING COUNT(DISTINCT od.BookID) >= 2;",

    // 14. Show the name of the customers who have ordered at least a book in category3 or category4 and the book names
    "SELECT DISTINCT c.FirstName, c.LastName, b.Title
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    JOIN db_book b ON od.BookID = b.BookID
    JOIN db_subject s ON b.SubjectID = s.SubjectID
    WHERE s.CategoryName IN ('category3', 'category4');",

    // 15. Show the name of the customer who has ordered at least one book written by author1
    "SELECT DISTINCT c.FirstName, c.LastName
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    JOIN db_book b ON od.BookID = b.BookID
    WHERE b.Author = 'author1';",

    // 16. Show the name and total sale (price of orders) of each employee
    "SELECT 
    e.FirstName, 
    e.LastName, 
    COALESCE(SUM(od.Quantity * b.UnitPrice), 0) AS TotalSales
FROM db_employee e
LEFT JOIN db_order o ON e.EmployeeID = o.EmployeeID
LEFT JOIN db_order_detail od ON o.OrderID = od.OrderID
LEFT JOIN db_book b ON od.BookID = b.BookID
GROUP BY e.EmployeeID, e.FirstName, e.LastName;",

    // 17. Show the book names and their respective quantities for open orders at midnight 08/04/2016
    "SELECT b.Title, od.Quantity
    FROM db_book b
    JOIN db_order_detail od ON b.BookID = od.BookID
    JOIN db_order o ON od.OrderID = o.OrderID
    WHERE o.ShippedDate IS NULL OR o.ShippedDate > '2016-08-04 00:00:00';",

    // 18. Show the names of customers who have ordered more than 1 book and the corresponding quantities. List the result in the descending quantity
    "SELECT c.FirstName, c.LastName, SUM(od.Quantity) AS TotalQuantity
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    GROUP BY c.CustomerID, c.FirstName, c.LastName
    HAVING COUNT(DISTINCT od.BookID) > 1
    ORDER BY TotalQuantity DESC;",

    // 19. Show the names of customers who have ordered more than 3 books and their respective telephone numbers
    "SELECT c.FirstName, c.LastName, c.Phone
    FROM db_customer c
    JOIN db_order o ON c.CustomerID = o.CustomerID
    JOIN db_order_detail od ON o.OrderID = od.OrderID
    GROUP BY c.CustomerID, c.FirstName, c.LastName, c.Phone
    HAVING COUNT(DISTINCT od.BookID) > 3;"
);
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #fff;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        .query-form {
            margin-bottom: 30px;
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 14px;
        }
        .buttons {
            margin-bottom: 20px;
        }
        .buttons button {
            padding: 8px 15px;
            margin-right: 10px;
            background-color: #007bff;
            border: none;
            color: white;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .buttons button:hover {
            background-color: #0056b3;
        }
        .buttons button[type="reset"] {
            background-color: #6c757d;
        }
        .buttons button[type="reset"]:hover {
            background-color: #5a6268;
        }
        .message {
            padding: 12px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 0 5px rgba(0,0,0,0.05);
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            color: #333;
            font-weight: bold;
        }
        .numbered-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
        }
        .numbered-buttons button {
            width: 40px;
            height: 40px;
            font-size: 16px;
            font-weight: bold;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: background-color 0.3s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Online Bookstore Database - Logan Bolton</h1>
        
        <div class="query-form">
            <div class="numbered-buttons">
                <?php for ($i = 1; $i <= count($numberedQueries); $i++): ?>
                    <button type="button" onclick="loadQuery(<?php echo $i-1; ?>)"><?php echo $i; ?></button>
                <?php endfor; ?>
            </div>
            
            <form method="post" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
                <textarea id="query-textarea" name="query" placeholder="Enter your SQL query here..."><?php echo htmlspecialchars($query); ?></textarea>
                <div class="buttons">
                    <button type="submit">Execute Query</button>
                    <button type="button" onclick="clearTextarea()">Clear</button>
                </div>
            </form>
            
        </div>
        
        <?php if (!empty($error)): ?>
            <div class="message error"><?php echo $error; ?></div>
        <?php endif; ?>
        
        <?php if (!empty($success)): ?>
            <div class="message success"><?php echo $success; ?></div>
        <?php endif; ?>
        
        <?php if ($result && $result->num_rows > 0): ?>
            <h2>Query Results</h2>
            <table>
                <thead>
                    <tr>
                        <?php
                        $fields = $result->fetch_fields();
                        foreach ($fields as $field) {
                            echo "<th>" . htmlspecialchars($field->name) . "</th>";
                        }
                        ?>
                    </tr>
                </thead>
                <tbody>
                    <?php
                    // Display query results
                    while ($row = $result->fetch_assoc()) {
                        echo "<tr>";
                        foreach ($row as $value) {
                            echo "<td>" . htmlspecialchars(($value !== null) ? $value : "NULL") . "</td>";
                        }
                        echo "</tr>";
                    }
                    ?>
                </tbody>
            </table>
            <p>Total rows: <?php echo $rowCount; ?></p>
        <?php endif; ?>
    </div>

    <script>
        function loadQuery(index) {
            var queries = <?php echo json_encode($numberedQueries); ?>;
            document.getElementById('query-textarea').value = queries[index];
        }

        function clearTextarea() {
            document.getElementById('query-textarea').value = '';
            
            var successMessage = document.querySelector('.message.success');
            var errorMessage = document.querySelector('.message.error');
            
            if (successMessage) {
                successMessage.style.display = 'none';
            }
            
            if (errorMessage) {
                errorMessage.style.display = 'none';
            }
        }
    </script>
</body>
</html>