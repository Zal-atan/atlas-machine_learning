-- lists all genres from hbtn_0d_tvshows and displays the number of shows linked to each.
SELECT TV_GENRES.name as genre, count(TV_SHOW_GENRES.show_id) as number_of_shows
FROM TV_GENRES
LEFT JOIN TV_SHOW_GENRES ON TV_GENRES.id = TV_SHOW_GENRES.genre_id
GROUP BY TV_GENRES.name
ORDER BY number_of_shows DESC;

