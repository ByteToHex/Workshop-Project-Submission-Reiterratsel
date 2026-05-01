from datetime import datetime, timedelta


'''
give me a sample MODULAR  py script to count

(1) total days from some period (Exact date- date to date)

(2) total weekdays in some period (as above)

sample can maybe do 2022-07-28 til 2025-12-10
'''

def get_total_days(start_date, end_date):
    """Calculates the absolute difference in days between two dates."""
    delta = end_date - start_date
    return abs(delta.days)

def get_total_weekdays(start_date, end_date):
    """Iterates through the range to count Monday through Friday."""
    count = 0
    current_date = min(start_date, end_date)
    final_date = max(start_date, end_date)
    
    while current_date <= final_date:
        # Monday is 0, Sunday is 6
        if current_date.weekday() < 5:
            count += 1
        current_date += timedelta(days=1)
    return count

def main():
    # Define date strings
    start_str = "2022-07-28"
    end_str = "2025-12-10"
    
    # Parse strings into datetime objects
    fmt = "%Y-%m-%d"
    start_dt = datetime.strptime(start_str, fmt)
    end_dt = datetime.strptime(end_str, fmt)
    
    # Execute calculations
    total = get_total_days(start_dt, end_dt)
    weekdays = get_total_weekdays(start_dt, end_dt)
    
    print(f"Period: {start_str} to {end_str}")
    print(f"Total Days: {total}")
    print(f"Total Weekdays: {weekdays}")

if __name__ == "__main__":
    main()