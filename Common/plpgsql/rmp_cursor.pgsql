create or replace function rmp_cursor(tbl text, v_cur refcursor) returns refcursor as
$$
begin
open v_cur for execute 'select * from ' || tbl::regclass;
return v_cur;
end;
$$ language plpgsql;
