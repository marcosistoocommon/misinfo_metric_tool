import {
  index,
  pgTable,
  text,
  uniqueIndex,
  varchar,
} from "drizzle-orm/pg-core";
import { timestamps } from "./utils";

export const users = pgTable(
  "users",
  {
    id: varchar("id", { length: 30 }).primaryKey(),
    email: text("email").notNull().unique(),
    imageUrl: text("image_url"),
    ...timestamps,
  },
  (table) => [
    uniqueIndex("users_email_idx").on(table.email),
    index("users_created_at_idx").on(table.createdAt),
  ]
);
