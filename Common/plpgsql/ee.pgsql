create or replace function ee(int, tbl text) returns setof text as $$
declare 
v_cur refcursor;
v_data text;
begin
raise notice 'tablename: %', $2;
open v_cur for execute 'select * from ' || tbl::regclass || ' limit ' || $1;
while true loop
fetch v_cur into v_data;
if found then 
return next v_data;
else
return;
end if;
end loop;
end;
$$ language plpgsql;
