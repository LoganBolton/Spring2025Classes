<?php
// Database configuration file - Placed directly in public_html directory

// Database connection parameters
$host = "sysmysql8.auburn.edu";
$username = "ldb0046"; // Your Auburn username
$password = "100Hunters!"; // Replace with your actual database password
$database = "ldb0046db"; // Your database name (usually username + "db")

/**
 * Establishes a database connection using mysqli
 * 
 * @return mysqli The database connection object
 */
function connectDB() {
    global $host, $username, $password, $database;
    
    // Create connection
    $conn = new mysqli($host, $username, $password, $database);
    
    // Check connection
    if ($conn->connect_error) {
        die("Connection failed: " . $conn->connect_error);
    }
    
    return $conn;
}
?>