<?php
$servername = "localhost";
$username = "bqmnhat2313";
$password = "Dongden@2313";
$database = "diagnosis";

$conn = new mysqli($servername, $username, $password, $database);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}
?>
