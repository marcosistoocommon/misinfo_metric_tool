CREATE TABLE
	"checks" (
		"id" varchar(30) PRIMARY KEY NOT NULL,
		"slug" varchar(100) NOT NULL,
		"user_id" varchar(30) NOT NULL,
		"text_id" varchar(30) NOT NULL,
		"result" jsonb,
		"status" varchar(20) DEFAULT 'pending' NOT NULL,
		"completed_at" timestamp,
		"created_at" timestamp DEFAULT now () NOT NULL,
		"updated_at" timestamp DEFAULT now () NOT NULL,
		CONSTRAINT "checks_slug_unique" UNIQUE ("slug")
	);

--> statement-breakpoint
CREATE TABLE
	"texts" (
		"id" varchar(30) PRIMARY KEY NOT NULL,
		"hash" varchar(64) NOT NULL,
		"content" text NOT NULL,
		"word_count" varchar(10),
		"created_at" timestamp DEFAULT now () NOT NULL,
		"updated_at" timestamp DEFAULT now () NOT NULL,
		CONSTRAINT "texts_hash_unique" UNIQUE ("hash")
	);

--> statement-breakpoint
ALTER TABLE "checks" ADD CONSTRAINT "checks_user_id_users_id_fk" FOREIGN KEY ("user_id") REFERENCES "public"."users" ("id") ON DELETE no action ON UPDATE no action;

--> statement-breakpoint
ALTER TABLE "checks" ADD CONSTRAINT "checks_text_id_texts_id_fk" FOREIGN KEY ("text_id") REFERENCES "public"."texts" ("id") ON DELETE no action ON UPDATE no action;

--> statement-breakpoint
CREATE INDEX "checks_user_id_idx" ON "checks" USING btree ("user_id");

--> statement-breakpoint
CREATE INDEX "checks_slug_idx" ON "checks" USING btree ("slug");

--> statement-breakpoint
CREATE INDEX "checks_text_id_idx" ON "checks" USING btree ("text_id");

--> statement-breakpoint
CREATE INDEX "checks_status_idx" ON "checks" USING btree ("status")
WHERE
	status = 'pending';

--> statement-breakpoint
CREATE INDEX "checks_created_at_idx" ON "checks" USING btree ("created_at");

--> statement-breakpoint
CREATE UNIQUE INDEX "texts_hash_idx" ON "texts" USING btree ("hash");

--> statement-breakpoint
CREATE INDEX "texts_word_count_idx" ON "texts" USING btree ("word_count");

--> statement-breakpoint
CREATE INDEX "texts_created_at_idx" ON "texts" USING btree ("created_at");