pub fn current_utc_timestamp() -> String {
    let duration = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format_utc_timestamp(duration.as_secs())
}

pub fn format_utc_timestamp(timestamp: u64) -> String {
    let second = timestamp % 60;
    let minute = (timestamp / 60) % 60;
    let hour = (timestamp / 3600) % 24;
    let mut days = timestamp / 86400;
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let month_days = month_lengths(is_leap_year(year));
    let mut month = 1u64;
    for &length in &month_days {
        if days < length {
            break;
        }
        days -= length;
        month += 1;
    }
    let day = days + 1;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

pub fn parse_timestamp_to_epoch(timestamp: &str) -> Option<u64> {
    let parts: Vec<&str> = timestamp.split('T').collect();
    if parts.len() != 2 {
        return None;
    }
    let date_parts: Vec<u64> = parts[0].split('-').filter_map(|p| p.parse().ok()).collect();
    if date_parts.len() != 3 {
        return None;
    }
    let time_str = parts[1].trim_end_matches('Z');
    let time_parts: Vec<u64> = time_str.split(':').filter_map(|p| p.parse().ok()).collect();
    if time_parts.len() != 3 {
        return None;
    }
    let (year, month, day) = (date_parts[0], date_parts[1], date_parts[2]);
    let (hour, minute, second) = (time_parts[0], time_parts[1], time_parts[2]);
    let mut total_days: u64 = 0;
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }
    let months = month_lengths(is_leap_year(year));
    for &month_days in months.iter().take(month.saturating_sub(1) as usize) {
        total_days += month_days;
    }
    total_days += day.saturating_sub(1);
    Some(total_days * 86400 + hour * 3600 + minute * 60 + second)
}

fn is_leap_year(year: u64) -> bool {
    (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400)
}

fn month_lengths(leap: bool) -> [u64; 12] {
    let feb = if leap { 29 } else { 28 };
    [31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
}
